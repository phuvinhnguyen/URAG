from systems.abstract import AbstractRAGSystem
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List, Optional
import re
import logging

logger = logging.getLogger(__name__)

# Use the same SYSTEM_PROMPT as other systems for consistency
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

class SelfLLMSystem(AbstractRAGSystem):
    """
    Self-RAG LLM system that handles model inference and probability calculation.
    
    This system provides the LLM interface for Self-RAG, handling:
    1. Model loading and inference
    2. Probability calculation for answer options
    3. Reflection token extraction from generated text
    
    Based on the original Self-RAG paper: https://arxiv.org/abs/2310.11511
    """
    
    def __init__(self, model_name: str = "selfrag/selfrag_llama2_7b", device: str = "cuda", 
                 technique: str = "selfrag", max_new_tokens: int = 300, temperature: float = 0.1,
                 method: str = 'normal', threshold: float = 0.2):
        self.device = device
        self.technique = technique
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.method = method
        self.threshold = threshold
        self.model_name = model_name
        
        print(f"[SELFRAG] Loading Self-RAG model: {model_name}")
        
        # Load Self-RAG model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            padding_side="left"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"[SELFRAG] Model loaded successfully. Vocab size: {len(self.tokenizer)}")
        
        # Self-RAG special tokens
        self.reflection_tokens = {
            'retrieve': '<|retrieve|>',
            'no_retrieve': '<|no_retrieve|>',
            'relevant': '<|relevant|>',
            'irrelevant': '<|irrelevant|>',
            'supported': '<|fully_supported|>',
            'partially_supported': '<|partially_supported|>',
            'no_support': '<|no_support|>',
            'utility_5': '<|utility:5|>',
            'utility_4': '<|utility:4|>',
            'utility_3': '<|utility:3|>',
            'utility_2': '<|utility:2|>',
            'utility_1': '<|utility:1|>'
        }
    
    def get_batch_size(self) -> int: return 40
    
    def _create_unified_prompt(self, system_message: str, user_message: str) -> str:
        """Create unified prompt following Self-RAG format."""
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
                tokenize=False, add_generation_prompt=True
            )
        except:
            # Fallback to Self-RAG format
            return f"### Instruction:\n{system_message}\n\n### Input:\n{user_message}\n\n### Response:"
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt for Self-RAG inference."""
        question = sample.get('question', '')
        context = sample.get('context', '')
        technique = sample.get('technique', self.technique)
        
        # Use standard SYSTEM_PROMPT for consistency with other systems
        if technique == 'cot':
            system_message = SYSTEM_PROMPT + " Think step by step and provide detailed reasoning."
            user_message = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        elif technique == 'rag' or technique == 'selfrag':
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
    
    def extract_reflection_tokens(self, text: str) -> Dict[str, Any]:
        """Extract reflection tokens from generated text."""
        tokens = {}
        
        # Check for retrieval decision
        if '<|retrieve|>' in text:
            tokens['retrieval'] = 'retrieve'
        elif '<|no_retrieve|>' in text:
            tokens['retrieval'] = 'no_retrieve'
            
        # Check for relevance assessment
        if '<|relevant|>' in text:
            tokens['relevance'] = 'relevant'
        elif '<|irrelevant|>' in text:
            tokens['relevance'] = 'irrelevant'
            
        # Check for support assessment
        if '<|fully_supported|>' in text:
            tokens['support'] = 'supported'
        elif '<|partially_supported|>' in text:
            tokens['support'] = 'partially_supported'
        elif '<|no_support|>' in text:
            tokens['support'] = 'no_support'
            
        # Check for utility rating
        for i in range(5, 0, -1):
            if f'<|utility:{i}|>' in text:
                tokens['utility'] = i
                break
                
        return tokens
    
    def make_retrieval_decision(self, question: str, options: List[str]) -> bool:
        """Helper method to determine if retrieval is needed based on reflection tokens."""
        # Create a simple prompt for retrieval decision with reflection tokens
        system_message = """You are a helpful assistant that decides whether to retrieve information. Use these reflection tokens:
- <|retrieve|> when you need external information to answer the question
- <|no_retrieve|> when you can answer the question directly with your knowledge

Respond with only the appropriate reflection token."""
        user_message = f"Question: {question}\nOptions: {', '.join(options)}\n\nDo you need to retrieve information to answer this question?"
        
        prompt = self._create_unified_prompt(system_message, user_message)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract reflection tokens
        reflection_tokens = self.extract_reflection_tokens(generated_text)
        
        # Return True if model decides to retrieve
        return reflection_tokens.get('retrieval') == 'retrieve'
    
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
    