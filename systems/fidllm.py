from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, Any, List, Tuple
import torch.nn.functional as F
from utils.fid import FiDGenerator
from haystack import Document
from haystack import Document
from typing import Dict, Any, List, Union

def build_fid_inputs(passages, prefix, choices) -> Dict[str, Any]:


    # 2) Chuẩn hóa passages -> documents (content phải là str)
    documents: List[Document] = []
    for i, p in enumerate(passages):
        p = "" if p is None else str(p).strip()
        if p:  # bỏ passage rỗng
            documents.append(Document(content=p, meta={"passage_id": i}))

    if isinstance(choices, dict):
        choices_text = "\n".join([f"{k}: {v}" for k, v in choices.items()])
    elif isinstance(choices, list):
        # hỗ trợ list ["A. ...", "B. ..."] hoặc [("A","..."), ...]
        if choices and isinstance(choices[0], (tuple, list)) and len(choices[0]) == 2:
            choices_text = "\n".join(f"{k}: {v}" for k, v in choices)
        else:
            choices_text = "\n".join(map(str, choices))
    else:
        choices_text = str(choices)

    # prefix của bạn đã có "Question: ..." và "Answer:" sẵn.
    # Ta chỉ nối Choices + yêu cầu output.
    prompt = (
        str(prefix).rstrip() +
        "\nChoices:\n" + choices_text +
        "\nAnswer with only the letter (A, B, C, or D): |Answer"
    )

    # Sanity check kiểu dữ liệu để tránh lỗi "concatenate str with list"
    assert isinstance(prompt, str), f"prompt phải là str, hiện là {type(prompt)}"
    assert all(isinstance(d.content, str) for d in documents), "Document.content phải là str"

    return {"prompt": prompt, "documents": documents}



class FiDLLMSystem(AbstractRAGSystem):
    """
    FiD LLM system that provides both regular T5 and FiD-trained T5 processing.
    
    This system provides two processing modes:
    1. Regular T5 model for standard text generation with probability calculation
    2. FiD-trained T5 model for fusion-in-decoder processing with document context using FiDConcatEncoder
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base", fid_model_name: str = "Intel/fid_t5_large_nq", device: str = "cuda", num_samples: int = 20, technique: str = "direct", temperature: float = 0.1, max_new_tokens: int = 64, method: str = 'normal'):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.technique = technique
        self.method = method
        self.model_name = model_name
        self.fid_model_name = fid_model_name
        self.fid_generator = FiDGenerator(
            model=fid_model_name,
            generation_kwargs={"max_length": 50, "do_sample": False, "temperature": 0.1},
            huggingface_pipeline_kwargs={"device_map": "auto"}
        )
        self.fid_generator.warm_up()
        # Load regular T5 model with error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Loaded regular T5 model: {model_name}")
        except Exception as e:
            print(f"✗ Failed to load regular T5 model {model_name}: {e}")
            # Fallback to a smaller, guaranteed available model
            fallback_model = "google/flan-t5-small"
            print(f"→ Falling back to {fallback_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, use_fast=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
            self.model.to(self.device)
            self.model.eval()
            self.model_name = fallback_model
        
        
    
    def get_batch_size(self) -> int: return 80
    
    def _extract_answer_from_response(self, response: str) -> str:
        """Extract answer from structured response format - returns only A, B, C, D, or E."""
        import re
        
        # Look for Answer|X pattern
        match = re.search(r'Answer\|([A-E])', response)
        if match:
            return match.group(1)
        
        # Fallback: look for single letter answers at the end or beginning
        response_clean = response.strip()
        if len(response_clean) == 1 and response_clean in 'ABCDE':
            return response_clean
        
        # Look for pattern like "A.", "B)", etc.
        match = re.search(r'\b([A-E])[.)]\s*$', response)
        if match:
            return match.group(1)
        
        # Last resort: return first letter found in ABCDE
        for char in response:
            if char in 'ABCDE':
                return char
        
        return 'A'  # Default fallback
    
    def _create_t5_prompt(self, sample: Dict[str, Any]) -> str:
        """Create prompt for regular T5 model with structured output format."""
        question = sample.get('question', '')
        context = sample.get('context', '')
        
        if self.technique == 'rag' and context:
            # For RAG, separate system instruction from context and question
            system_instruction = "Answer the multiple choice question using the provided context. You must respond with ONLY one letter: A, B, C, D,.... Provide your final answer in the format Answer|X where X is one of A, B, C, D,..."
            prompt = f"{system_instruction}\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:"
        else:
            # For direct, include system instruction with question
            system_instruction = "Answer the multiple choice question. You must respond with ONLY one letter: A, B, C, D,.... Provide your final answer in the format Answer|X where X is one of A, B, C, D,..."
            prompt = f"{system_instruction}\n\nQuestion: {question}\n\nAnswer:"
        
        return prompt
    
    def _create_fid_passages_and_prefix(self, sample: Dict[str, Any]) -> Tuple[List[str], str]:
        """Create passages and prefix for FiD encoder processing."""
        question = sample.get('question', '')
        context = sample.get('context', '')
        
        # Create prefix (instruction + question)
        prefix = f"""Answer the multiple choice question using the provided context.
        You must respond with ONLY one letter: A, B, C, D,....
        Provide your final answer in the format Answer|X where X is one of A, B, C, D,...
        Question: {question}
        Answer:"""
        if context:
            # Split context by bullet points or newlines to create multiple passages
            passages = [doc.strip() for doc in context.split('\n-') if doc.strip()]
            if not passages:
                passages = [context]
        else:
            passages = [""]
        
        return passages, prefix
    
    def _generate_with_probabilities_t5(self, prompt: str, options: List[str]):
        """Generate response using regular T5 with probability calculation."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                num_beams=1
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Calculate probabilities for options using a more robust approach
        conformal_probabilities = {}
        if len(outputs.scores) > 0 and options:
            # For T5, we need to look at multiple tokens to find the answer
            # Get all token logits from the generation
            all_token_logits = outputs.scores  # List of tensors, one per generated token
            
            # Get token IDs for options
            option_token_ids = {}
            for option in options:
                option_ids = self.tokenizer.encode(option, add_special_tokens=False)
                if len(option_ids) > 0:
                    option_token_ids[option] = option_ids[0]
                else:
                    option_token_ids[option] = self.tokenizer.unk_token_id
            
            # Find the best matching token across all generated positions
            option_scores = {option: float('-inf') for option in options}
            
            for token_logits in all_token_logits:
                token_probs = F.softmax(token_logits[0], dim=-1)
                for option, token_id in option_token_ids.items():
                    current_score = token_probs[token_id].item()
                    option_scores[option] = max(option_scores[option], current_score)
            
            # Normalize scores to probabilities
            total_score = sum(max(0, score) for score in option_scores.values())
            if total_score > 0:
                conformal_probabilities = {
                    option: max(0, score) / total_score 
                    for option, score in option_scores.items()
                }
            else:
                # Fallback to uniform
                uniform_prob = 1.0 / len(options)
                conformal_probabilities = {option: uniform_prob for option in options}
        else:
            # Fallback to uniform probabilities
            uniform_prob = 1.0 / len(options) if options else 1.0
            conformal_probabilities = {option: uniform_prob for option in options}
        
        return generated_text, conformal_probabilities
    
  
    def _generate_with_probabilities_fid_generator(self, passages: List[str], prefix: str, options: List[str]):
        """Generate response using FiD generator with probability calculation."""
        inputs = build_fid_inputs(passages, prefix, options)
        result = self.fid_generator.run(
            prompt=inputs["prompt"],
            documents=inputs["documents"]  # List[Document]
        )
        generated_response = result["replies"]
        scores = result["scores"]
        conformal_probabilities = {}
        if len(scores) > 0 and options:
            all_token_logits = scores
            
            # Get token IDs for options
            # THAY ĐỔI: self.fid_encoder.tokenizer → fid_generator.pipeline.tokenizer
            option_token_ids = {}
            for option in options:
                option_ids = self.fid_generator.pipeline.tokenizer.encode(option, add_special_tokens=False)
                if len(option_ids) > 0:
                    option_token_ids[option] = option_ids[0]
                else:
                    option_token_ids[option] = self.fid_generator.pipeline.tokenizer.unk_token_id
            
            # Find the best matching token across all generated positions
            option_scores = {option: float('-inf') for option in options}
            
            for token_logits in all_token_logits:
                token_probs = F.softmax(token_logits[0], dim=-1)
                for option, token_id in option_token_ids.items():
                    current_score = token_probs[token_id].item()
                    option_scores[option] = max(option_scores[option], current_score)
            
            # Normalize scores to probabilities
            total_score = sum(max(0, score) for score in option_scores.values())
            if total_score > 0:
                conformal_probabilities = {
                    option: max(0, score) / total_score 
                    for option, score in option_scores.items()
                }
            else:
                uniform_prob = 1.0 / len(options)
                conformal_probabilities = {option: uniform_prob for option in options}
        else:
            uniform_prob = 1.0 / len(options) if options else 1.0
            conformal_probabilities = {option: uniform_prob for option in options}
        
        return generated_response, conformal_probabilities
    
    def _generate_response_with_probabilities(self, prompt: str, options: List[str]):
        """Override base class method to handle both regular T5 and FiD processing."""
        import json
        
        # Check if prompt contains FiD data (passages and prefix)
        try:
            # Try to parse as JSON - if successful, it's FiD data
            fid_data = json.loads(prompt)
            if 'passages' in fid_data and 'prefix' in fid_data:
                # FiD processing
                passages = fid_data['passages']
                prefix = fid_data['prefix']
                response, conformal_probabilities = self._generate_with_probabilities_fid_generator(passages, prefix, options)
                
                # Force answer to be only A, B, C, D, or E
                clean_answer = self._extract_answer_from_response(response)
                
                return response, conformal_probabilities
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Regular T5 processing
        response, conformal_probabilities = self._generate_with_probabilities_t5(prompt, options)
        
        return response, conformal_probabilities

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample using regular T5 model."""
        prompt = self._create_t5_prompt(sample)
        options = sample.get('options', [])
        response, conformal_probabilities = self._generate_response_with_probabilities(prompt, options)
        
           
        return {
            'id': sample.get('id', 'unknown'),
            'prompt': prompt,
            'generated_response': response,
            'predicted_answer': max(conformal_probabilities.items(), key=lambda x: x[1])[0],
            'conformal_probabilities': conformal_probabilities,
            'question': sample.get('question', ''),
            'context': sample.get('context', ''),
            'full_response_data': {
                'model_response': response,
                'input_prompt': prompt
            }
        }
    
    def process_sample_fid(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample using FiD-trained T5 model with improved passage handling."""
        import json
        
        passages, prefix = self._create_fid_passages_and_prefix(sample)
        options = sample.get('options', [])
        
        # Create FiD prompt as JSON string for the base class method
        fid_prompt = json.dumps({
            'passages': passages,
            'prefix': prefix
        })
        
        response, conformal_probabilities = self._generate_response_with_probabilities(fid_prompt, options)

        return {
            'id': sample.get('id', 'unknown'),
            'generated_response': response,
            'predicted_answer': max(conformal_probabilities.items(), key=lambda x: x[1])[0],
            'conformal_probabilities': conformal_probabilities,
            'passages': passages,
            'fid_prefix': prefix,
            'question': sample.get('question', ''),
            'context': sample.get('context', '')
        }
    
