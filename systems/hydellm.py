from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F
from typing import Dict, Any, List

class EndSequenceCriteria(StoppingCriteria):
    def __init__(self, end_ids):
        super().__init__()
        self.end_ids = end_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if len(input_ids[0]) < len(self.end_ids) + 1: return False
        return torch.equal(input_ids[0][-len(self.end_ids)-1:-1], torch.tensor(self.end_ids, device=input_ids.device))

class HyDELLMSystem(AbstractRAGSystem):
    """
    HyDE LLM system that generates hypothetical documents for better retrieval.
    
    This system first generates a hypothetical document that would answer the query,
    then uses that document for retrieval instead of the original query.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda", num_samples: int = 20, technique: str = "direct", temperature: float = 0.1, max_new_tokens: int = 1024):
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.technique = technique
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, device_map="auto" if self.device == "cuda" else None)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_batch_size(self) -> int: return 1
    
    def _create_unified_prompt(self, system_message: str, user_message: str) -> str:
        """Create a unified prompt format that works across different model types."""
        # Detect model type and format accordingly
        model_lower = self.model_name.lower()
        
        # Llama-style models (Llama, Code Llama, etc.)
        if "llama" in model_lower:
            return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]\n\n"
        
        # Mistral models
        elif "mistral" in model_lower:
            return f"<s>[INST] {system_message}\n\n{user_message} [/INST]"
        
        # Falcon models
        elif "falcon" in model_lower:
            return f"User: {system_message}\n\n{user_message}\n\nAssistant:"
        
        # MPT models
        elif "mpt" in model_lower:
            return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        # DialoGPT and GPT-style models
        elif any(x in model_lower for x in ["dialogpt", "gpt"]):
            return f"{system_message}\n\n{user_message}\n\nResponse:"
        
        # Default fallback format
        else:
            return f"### Instruction:\n{system_message}\n\n### Input:\n{user_message}\n\n### Response:"
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample.get('question', '')
        context = sample.get('context', '')
        
        # Generate unified prompts based on technique
        if self.technique == 'cot':
            system_message = "You are a helpful assistant that answers multiple choice questions with step-by-step reasoning. Think through the problem carefully and provide your final answer in the format Answer|X where X is one of A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts."
            user_message = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is one of A, B, C, D, ..."
        elif self.technique == 'rag' or self.technique == 'hyde':
            if context:
                system_message = "You are a helpful assistant that answers multiple choice questions using the provided context information. Use the context to inform your answer and provide the final answer in the format Answer|X where X is one of A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts."
                user_message = f"Context information: {context}\n\nQuestion: {question}\n\nPlease provide your final answer in the format Answer|X where X is one of A, B, C, D, ..."
            else:
                system_message = "You are a helpful assistant that answers multiple choice questions. Provide your final answer in the format Answer|X where X is one of A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts."
                user_message = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is one of A, B, C, D, ..."
        else:
            # Direct prompting
            system_message = "You are a helpful assistant that answers multiple choice questions. Provide your final answer in the format Answer|X where X is one of A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts."
            user_message = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is one of A, B, C, D, ..."
        
        try:
            prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
                ], tokenize=False, add_generation_prompt=True)
        except:
            prompt = self._create_unified_prompt(system_message, user_message)
        
        return prompt

    def _generate_response_with_probabilities(self, prompt: str, options: List[str] = None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start_ids = self.tokenizer.encode("Answer|", add_special_tokens=False)
        stopping = StoppingCriteriaList([EndSequenceCriteria(start_ids)])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
                return_dict_in_generate=True,
                output_scores=True,
                repetition_penalty=1.1,
                top_p=0.9,
                top_k=40,
                num_return_sequences=1
            )

        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        if options is not None:
            last_token_probs = F.softmax(outputs.scores[-1][0], dim=-1)
            option_tokens = [self.tokenizer.encode(option, add_special_tokens=False)[0] for option in options]
            last_token_probs = last_token_probs[option_tokens] + 1e-10
            last_token_probs = last_token_probs / last_token_probs.sum()
            conformal_probabilities = {option: last_token_probs[i].item() for i, option in enumerate(options)}
        else:
            conformal_probabilities = {}
        
        return generated_text, conformal_probabilities
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._generate_prompt(sample)
        
        options = sample.get('options', [])

        response, conformal_probabilities = self._generate_response_with_probabilities(prompt, options)
        predicted_answer = max(conformal_probabilities.items(), key=lambda x: x[1])[0]
        
        return {
            'id': sample.get('id', 'unknown'),
            'generated_response': response,
            'predicted_answer': predicted_answer,
            'conformal_probabilities': conformal_probabilities,
            'technique': self.technique,
        }
