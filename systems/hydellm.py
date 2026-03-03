from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any

class HyDELLMSystem(AbstractRAGSystem):
    """
    HyDE LLM system that generates hypothetical documents for better retrieval.
    
    This system first generates a hypothetical document that would answer the query,
    then uses that document for retrieval instead of the original query.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda", num_samples: int = 20, technique: str = "direct", temperature: float = 0.1, max_new_tokens: int = 1024, method: str = 'normal', **kwargs):
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.technique = technique
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device == "cuda" else None)
        self.model.eval()
        self.model_name = model_name
        self.method = method
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_batch_size(self) -> int: return 40
    
    def _create_unified_prompt(self, system_message: str, user_message: str) -> str:
        try:
            return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
                ], tokenize=False, add_generation_prompt=True)
        except:
            model_lower = self.model_name.lower()
            
            if "llama" in model_lower: return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]\n\n"
            elif "mistral" in model_lower: return f"<s>[INST] {system_message}\n\n{user_message} [/INST]"
            elif "falcon" in model_lower: return f"User: {system_message}\n\n{user_message}\n\nAssistant:"
            elif "mpt" in model_lower: return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            elif any(x in model_lower for x in ["dialogpt", "gpt"]): return f"{system_message}\n\n{user_message}\n\nResponse:"
            else: return f"### Instruction:\n{system_message}\n\n### Input:\n{user_message}\n\n### Response:"
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample.get('question', '')
        
        if self.technique == 'cot':
            system_message = "You are a helpful assistant that answers multiple choice questions with step-by-step reasoning. Think through the problem carefully and provide your final answer in the format Answer|X where X is one of A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts."
            user_message = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is one of A, B, C, D, ..."
        elif self.technique == 'rag' or self.technique == 'hyde':
            context = sample.get('context', '')
            if context:
                system_message = "You are a helpful assistant that answers multiple choice questions using the provided context information. Use the context to inform your answer and provide the final answer in the format Answer|X where X is one of A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts."
                user_message = f"Context information: {context}\n\nQuestion: {question}\n\nPlease provide your final answer in the format Answer|X where X is one of A, B, C, D, ..."
            else:
                system_message = "You are a helpful assistant that answers multiple choice questions. Provide your final answer in the format Answer|X where X is one of A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts."
                user_message = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is one of A, B, C, D, ..."
        else:
            system_message = "You are a helpful assistant that answers multiple choice questions. Provide your final answer in the format Answer|X where X is one of A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts."
            user_message = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is one of A, B, C, D, ..."
        
        return self._create_unified_prompt(system_message, user_message)
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._generate_prompt(sample)
        options = sample.get('options', [])
        response, conformal_probabilities = self._generate_response_with_probabilities(prompt, options)
        return {
            'id': sample.get('id', 'unknown'),
            'generated_response': response,
            'predicted_answer': max(conformal_probabilities.items(), key=lambda x: x[1])[0],
            'conformal_probabilities': conformal_probabilities,
            'technique': self.technique,
        }
