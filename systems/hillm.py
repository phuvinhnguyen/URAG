from systems.abstract import AbstractRAGSystem
import os
import logging
import numpy as np
import httpx
import asyncio
from typing import Dict, Any, List

SYSTEM_PROMPT = """You are provided with a multiple choice question and various references. Your task is to answer the question succinctly, using format Answer|X (mandatory) where X is your answer for the multiple choice question, which can be A, B, C, D, ... Follow this format strictly because it is the only way to extract your inner thoughts. You must provide the final answer and finish the chat as soon as possible.
For example:
Question: What is the capital of France?
A. London
B. Berlin
C. Paris
D. Madrid
Answer|A

Question: What is the result of 2 + 2?
A. 3
B. 4
C. 5
D. 6
Answer|B
"""

class HiLLMSystem(AbstractRAGSystem):
    """
    HiLLM system using vLLM backend for inference - serves as a baseline without RAG.
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 model: str = None,  # Support both model_name and model for compatibility
                 technique: str = "direct", 
                 max_new_tokens: int = 100, 
                 temperature: float = 0.1, 
                 method: str = 'normal',
                 vllm_host: str = "http://localhost:8000/v1",
                 vllm_port: int = 8000,
                 vllm_base_url: str = None,
                 vllm_api_key: str = "123",
                 **kwargs):
        self.technique = technique
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.method = method
        
        # Use model parameter if provided, otherwise use model_name
        self.model_name = model or model_name
        
        # Configure vLLM connection
        # Handle case where vllm_host already includes full URL
        if vllm_host.startswith('http'):
            self.vllm_base_url = vllm_host
        else:
            self.vllm_base_url = vllm_base_url or f"http://{vllm_host}:{vllm_port}/v1"
        
        self.vllm_api_key = vllm_api_key
        
    def get_batch_size(self) -> int: 
        return 1
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt based on sample technique."""
        question = sample.get('question', '')
        
        if self.technique == 'cot': 
            prompt = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        elif self.technique == 'rag': 
            prompt = f"Context information: {sample.get('context', '')}\n\nQuestion: {question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        else: 
            prompt = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        
        return prompt

    async def _vllm_generate_response(self, prompt: str) -> str:
        """Generate response using vLLM HTTP API"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_new_tokens,
                    "stream": False
                }
                
                response = await client.post(
                    f"{self.vllm_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.vllm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content
                
            except Exception as e:
                logging.error(f"vLLM API call failed: {e}")
                raise

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._generate_prompt(sample)
        options = sample.get('options', [])
        
        # Generate response using async vLLM call
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        response = loop.run_until_complete(self._vllm_generate_response(prompt))
        
        # Extract answer from response
        predicted_answer = "A"  # Default fallback
        if "Answer|" in response:
            try:
                predicted_answer = response.split("Answer|")[1].strip()[0]
            except:
                pass
        
        # Create uniform probabilities for options (since we don't have logprobs from simple HTTP call)
        uniform_prob = 1.0 / len(options) if options else 1.0
        conformal_probabilities = {option: uniform_prob for option in options} if options else {"A": 1.0}
        
        # Boost probability for predicted answer
        if predicted_answer in conformal_probabilities:
            conformal_probabilities[predicted_answer] = 0.7
            remaining_prob = 0.3 / (len(options) - 1) if len(options) > 1 else 0.3
            for key in conformal_probabilities:
                if key != predicted_answer:
                    conformal_probabilities[key] = remaining_prob
        
        return {
            'id': sample.get('id', 'unknown'),
            'generated_response': response,
            'predicted_answer': predicted_answer,
            'conformal_probabilities': conformal_probabilities,
            'technique': self.technique
        }
