from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import re
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from collections import Counter

class SimpleLLMSystem(AbstractRAGSystem):
    """
    Simple LLM system without RAG - just direct prompting.
    
    This serves as a baseline and example implementation.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto", num_samples: int = 20, technique: str = "direct"):
        """Initialize the LLM system."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.num_samples = num_samples  # Số lần chạy model khi options rỗng
        self.technique = technique

        logger.info(f"Loading model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_batch_size(self) -> int:
        """Return batch size of 1 for this simple implementation."""
        return 1
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt based on sample technique."""
        question = sample.get('question', '')
        technique = self.technique
        
        if technique == 'cot':
            return f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format <answer>X</answer> where X is your answer."
        elif technique == 'rag':
            # For simple LLM, just mention if context exists but don't use it effectively
            context = sample.get('context', '')
            if context:
                return f"Context information: {context}\n\nQuestion: {question}\n\nPlease provide your final answer in the format <answer>X</answer>."
            else:
                return f"{question}\n\nPlease provide your final answer in the format <answer>X</answer> where X is your answer."
        else:
            # Direct prompting
            return f"{question}\n\nPlease provide your final answer in the format <answer>X</answer> where X is your answer."
    
    def _generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate response from the LLM."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _find_answer_positions(self, text: str) -> Tuple[int, int]:
        """Find the start and end positions of <answer>...</answer> tags."""
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.start(1), match.end(1)
        return -1, -1
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        start_pos, end_pos = self._find_answer_positions(response)
        if start_pos != -1:
            return response[start_pos:end_pos].strip()
        return "Unknown"
    
    def _generate_multiple_responses(self, prompt: str, num_samples: int) -> List[str]:
        """Generate multiple responses for empty options case."""
        responses = []
        for i in range(num_samples):
            # Sử dụng temperature cao hơn để tạo đa dạng
            response = self._generate_response(prompt, temperature=0.8)
            answer = self._extract_answer(response)
            if answer != "Unknown":
                responses.append(answer)
            logger.debug(f"Sample {i+1}/{num_samples}: {answer}")
        return responses
    
    def _compute_probabilities_from_samples(self, answers: List[str]) -> Tuple[Dict[str, float], List[str]]:
        """Compute probabilities from multiple answer samples."""
        if not answers:
            return {}, []
        
        # Đếm tần suất xuất hiện của mỗi answer
        answer_counts = Counter(answers)
        total_count = len(answers)
        
        # Tính xác suất
        probabilities = {}
        unique_options = list(answer_counts.keys())
        
        for answer, count in answer_counts.items():
            probabilities[answer] = count / total_count
        
        logger.info(f"Generated {len(unique_options)} unique options from {total_count} samples: {answer_counts}")
        
        return probabilities, unique_options
    
    def _compute_option_probabilities(self, response: str, options: List[str]) -> Dict[str, float]:
        """Compute softmax probabilities for each option."""
        start_pos, end_pos = self._find_answer_positions(response + '<answer>A</answer>')
        
        if start_pos == -1:
            # If no answer format found, return uniform distribution
            uniform_prob = 1.0 / len(options)
            return {option: uniform_prob for option in options}
        
        inputs = self.tokenizer.encode(response[:start_pos], return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            logits = self.model(inputs).logits[0, -1, :]

        option_tokens = [self.tokenizer.encode(option, add_special_tokens=False)[0] for option in options]

        logits = F.softmax(logits[option_tokens], dim=0)

        return {option: logits[i].item() for i, option in enumerate(options)}
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample through the LLM system."""
        # Generate prompt
        prompt = self._generate_prompt(sample)
        
        # Extract options from the sample
        options = sample.get('options', [])
        
        if not options:
            # Case 1: Empty options - generate multiple responses and compute frequency-based probabilities
            logger.info(f"Empty options detected. Generating {self.num_samples} responses...")
            
            # Generate multiple responses
            answers = self._generate_multiple_responses(prompt, self.num_samples)
            
            # Compute probabilities from frequency
            option_probabilities, generated_options = self._compute_probabilities_from_samples(answers)
            
            # Get the most frequent answer as predicted answer
            if option_probabilities:
                predicted_answer = max(option_probabilities.items(), key=lambda x: x[1])[0]
            else:
                predicted_answer = "Unknown"
            
            # Generate one final response for display
            final_response = self._generate_response(prompt, temperature=0.1)
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': final_response,
                'predicted_answer': predicted_answer,
                'option_probabilities': option_probabilities,
                'num_samples_generated': len(answers),
                'prompt_used': prompt,
                'technique': self.technique,
                'method': 'frequency_based'
            }
        else:
            # Case 2: Options provided - use original logit-based method
            logger.info(f"Using provided options: {options}")
            
            # Generate response
            response = self._generate_response(prompt, temperature=0.1)
            
            # Compute option probabilities using logits
            option_probabilities = self._compute_option_probabilities(response, options)
            
            # Extract the predicted answer
            predicted_answer = max(option_probabilities.items(), key=lambda x: x[1])[0]
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': response,
                'predicted_answer': predicted_answer,
                'option_probabilities': option_probabilities,
                'provided_options': options,
                'prompt_used': prompt,
                'technique': sample.get('technique', 'direct'),
                'method': 'logit_based'
            }