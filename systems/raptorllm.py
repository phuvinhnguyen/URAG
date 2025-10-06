from systems.abstract import AbstractRAGSystem
from systems.simplellm import SYSTEM_PROMPT
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
from pathlib import Path
import os

try:
    from raptor.raptor import RetrievalAugmentation, RetrievalAugmentationConfig
    from raptor.raptor import BaseQAModel, BaseSummarizationModel, BaseEmbeddingModel
    from sentence_transformers import SentenceTransformer
    RAPTOR_AVAILABLE = True
except ImportError:
    RAPTOR_AVAILABLE = False

class LocalEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def create_embedding(self, text: str):
        return self.model.encode(text)

class LocalQAModel(BaseQAModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def answer_question(self, context: str, question: str, max_tokens: int = 150) -> str:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=inputs.shape[1] + max_tokens, 
                                        temperature=0.7, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

class LocalSummarizationModel(BaseSummarizationModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        prompt = f"Summarize the following text:\n\n{context}\n\nSummary:"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=inputs.shape[1] + max_tokens,
                                        temperature=0.7, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        
        summary = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return summary.strip()

class RaptorLLMSystem(AbstractRAGSystem):
    """
    Simplified RAPTOR LLM system following SimpleLLMSystem patterns.
    Inherits from AbstractRAGSystem and adds RAPTOR tree capabilities.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda", 
                 technique: str = "direct", max_new_tokens: int = 100, temperature: float = 0.1, 
                 method: str = 'normal', **kwargs):
        
        # Initialize LLM (following SimpleLLMSystem pattern)
        self.device = device
        self.technique = technique
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.method = method
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if RAPTOR_AVAILABLE:
            # Create RAPTOR models using the shared LLM
            self.qa_model = LocalQAModel(self.model, self.tokenizer)
            self.summarization_model = LocalSummarizationModel(self.model, self.tokenizer)
            self.embedding_model = LocalEmbeddingModel()
            
            # Initialize RAPTOR configuration
            self.config = RetrievalAugmentationConfig(
                qa_model=self.qa_model,
                summarization_model=self.summarization_model,
                embedding_model=self.embedding_model
            )
            
            # Create RAPTOR system (no tree persistence)
            self.raptor = RetrievalAugmentation(config=self.config)
            self.tree_built = False
        else:
            self.raptor = None
    
    def get_batch_size(self) -> int: return 40
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt based on sample technique using SYSTEM_PROMPT."""
        question = sample.get('question', '')
        
        if self.technique == 'cot': 
            prompt = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        elif self.technique == 'rag': 
            prompt = f"Context information: {sample.get('context', '')}\n\nQuestion: {question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        else: 
            prompt = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        
        try:
            prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
                ], tokenize=False, add_generation_prompt=True)
        except: 
            pass

        return prompt
    
    def _build_tree_if_needed(self, sample: Dict[str, Any]):
        """Build RAPTOR tree from context if available and not already built."""
        if not RAPTOR_AVAILABLE or self.tree_built:
            return
        
        context = sample.get('search_results', sample.get('context', ''))
        if context and isinstance(context, str) and len(context.strip()) > 50:
            try:
                self.raptor.add_documents(context)
                self.tree_built = True
            except Exception:
                pass  # Fallback to normal LLM processing
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with RAPTOR enhancement if available, fallback to normal LLM."""
        
        # Try to build tree from context
        self._build_tree_if_needed(sample)
        
        # Try RAPTOR question answering first
        if RAPTOR_AVAILABLE and self.raptor and hasattr(self.raptor, 'tree') and self.raptor.tree:
            try:
                question = sample.get('question', '')
                raptor_answer = self.raptor.answer_question(question=question)
                
                if raptor_answer and len(raptor_answer.strip()) > 0:
                    # Use RAPTOR answer as context for final response generation
                    enhanced_sample = sample.copy()
                    enhanced_sample['context'] = f"RAPTOR Analysis: {raptor_answer}"
                    enhanced_sample['technique'] = 'rag'  # Use RAG technique for proper formatting
                    
                    # Generate response using AbstractRAGSystem methods with SYSTEM_PROMPT
                    prompt = self._generate_prompt(enhanced_sample)
                    options = enhanced_sample.get('options', [])
                    response, conformal_probabilities = self._generate_response_with_probabilities(prompt, options)
                    
                    return {
                        'id': enhanced_sample.get('id', 'unknown'),
                        'generated_response': response,
                        'predicted_answer': max(conformal_probabilities.items(), key=lambda x: x[1])[0],
                        'conformal_probabilities': conformal_probabilities,
                        'technique': 'rag',
                        'raptor_used': True,
                        'raptor_answer': raptor_answer
                    }
            except Exception:
                pass  # Fallback to normal processing
        
        # Fallback to normal LLM processing using AbstractRAGSystem methods
        prompt = self._generate_prompt(sample)
        options = sample.get('options', [])
        response, conformal_probabilities = self._generate_response_with_probabilities(prompt, options)
        
        return {
            'id': sample.get('id', 'unknown'),
            'generated_response': response,
            'predicted_answer': max(conformal_probabilities.items(), key=lambda x: x[1])[0],
            'conformal_probabilities': conformal_probabilities,
            'technique': self.technique,
            'raptor_used': False
        }
