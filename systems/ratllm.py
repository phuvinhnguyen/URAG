from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List
import re

SYSTEM_PROMPT = """You are provided with a multiple choice question and various references. Your task is to answer the question succinctly, using format Answer|X (mandatory) where X is your answer for the multiple choice question, which can be A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts. You must provide the final answer and finish the chat as soon as possible.

For example:
Question: What is the capital of France?
A. London
B. Berlin
C. Paris
D. Madrid
Answer|C

Question: What is the result of 2 + 2?
A. 3
B. 4
C. 5
D. 6
Answer|B
"""

class RATLLMSystem(AbstractRAGSystem):
    """
    RAT (Retrieval Augmented Thoughts) LLM system that supports chain-of-thought reasoning.
    
    This system can be used standalone or as part of RATRAGSystem for retrieval-augmented reasoning.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda", 
                 technique: str = "direct", max_new_tokens: int = 512, temperature: float = 0.1, 
                 method: str = 'normal'):
        self.device = device
        self.technique = technique
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.method = method
        self.model_name = model_name
        
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
    
    def get_batch_size(self) -> int: return 20
    
    def _create_unified_prompt(self, system_message: str, user_message: str) -> str:
        """Create unified prompt following the same pattern as other LLM systems."""
        try:
            return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ], tokenize=False, add_generation_prompt=True)
        except:
            model_lower = self.model_name.lower()
            
            if "llama" in model_lower:
                return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]\n\n"
            elif "mistral" in model_lower:
                return f"<s>[INST] {system_message}\n\n{user_message} [/INST]"
            elif "falcon" in model_lower:
                return f"User: {system_message}\n\n{user_message}\n\nAssistant:"
            elif "mpt" in model_lower:
                return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            elif any(x in model_lower for x in ["dialogpt", "gpt"]):
                return f"{system_message}\n\n{user_message}\n\nResponse:"
            else:
                return f"### Instruction:\n{system_message}\n\n### Input:\n{user_message}\n\n### Response:"
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample.get('question', '')
        
        if self.technique == 'cot':
            system_message = SYSTEM_PROMPT + " Think step by step and provide detailed reasoning."
            user_message = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        elif self.technique == 'rag' or self.technique == 'rat':
            context = sample.get('context', '')
            if context:
                system_message = SYSTEM_PROMPT + " Use the provided context to inform your answer."
                user_message = f"Context information: {context}\n\nQuestion: {question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
            else:
                system_message = SYSTEM_PROMPT
                user_message = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        else:
            system_message = SYSTEM_PROMPT
            user_message = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        
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
            'technique': self.technique
        }

    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompts = [self._generate_prompt(sample) for sample in samples]
        options = [sample.get('options', []) for sample in samples]
        responses, conformal_probabilities = self._generate_response_with_probabilities(prompts, options)
        return [{
            'id': sample.get('id', 'unknown'),
            'generated_response': response,
            'predicted_answer': max(conformal_probabilities[i].items(), key=lambda x: x[1])[0],
            'conformal_probabilities': conformal_probabilities[i],
            'technique': self.technique
        } for i, (response, sample) in enumerate(zip(responses, samples))]