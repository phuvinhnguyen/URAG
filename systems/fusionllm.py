from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import re
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from collections import Counter

class FusionLLMSystem(AbstractRAGSystem):
    """
    Fusion LLM system that generates multiple diverse queries for better retrieval.
    
    This system implements the query fusion technique:
    1. Generate multiple diverse queries from the original question
    2. Use these queries for enhanced retrieval
    3. Apply Reciprocal Rank Fusion to combine results
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", num_samples: int = 20, num_queries: int = 3):
        """Initialize the Fusion LLM system."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.num_samples = num_samples
        self.num_queries = num_queries  # Number of diverse queries to generate
            
        logger.info(f"Loading Fusion model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_batch_size(self) -> int:
        """Return batch size of 1 for this implementation."""
        return 1
    
    def generate_diverse_queries(self, question: str) -> List[str]:
        """Generate multiple diverse queries from the original question."""
        queries = [question]  # Include original question
        
        # Generate diverse query prompts
        query_prompts = [
            f"Rephrase this question in a different way: {question}\n\nRephrased question:",
            f"What is another way to ask about: {question}\n\nAlternative question:",
            f"Generate a related question that might help answer: {question}\n\nRelated question:",
            f"Create a more specific version of this question: {question}\n\nSpecific question:",
            f"What broader question encompasses: {question}\n\nBroader question:"
        ]
        
        for i, prompt in enumerate(query_prompts[:self.num_queries-1]):  # -1 because we already have original
            try:
                generated_query = self._generate_response(prompt, max_length=100, temperature=0.8)
                
                # Clean up the generated query
                generated_query = generated_query.split('\n')[0].strip()
                if generated_query and generated_query not in queries:
                    queries.append(generated_query)
                    logger.debug(f"Generated query {i+1}: {generated_query}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate query {i+1}: {e}")
                continue
        
        logger.info(f"Generated {len(queries)} diverse queries (including original)")
        return queries
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt based on sample technique."""
        question = sample.get('question', '')
        technique = sample.get('technique', 'direct')
        
        # Build options text if available
        options_text = ""
        if 'options' in sample and 'option_texts' in sample:
            options_text = "\n\nOptions:\n"
            for option in sample['options']:
                text = sample['option_texts'].get(option, option)
                options_text += f"{option}. {text}\n"
            options_text += "\nPlease choose one of the options A, B, C, or D."
        
        if technique == 'cot':
            return f"Let's think step by step.\n\n{question}{options_text}\n\nPlease provide your reasoning and then give your final answer in the format <answer>X</answer> where X is one of A, B, C, or D."
        elif technique == 'rag' or technique == 'fusion':
            # For Fusion, use context if available
            context = sample.get('search_results', sample.get('context', ''))
            if context:
                return f"Context information: {context}\n\nQuestion: {question}{options_text}\n\nPlease provide your final answer in the format <answer>X</answer> where X is one of A, B, C, or D."
            else:
                return f"{question}{options_text}\n\nPlease provide your final answer in the format <answer>X</answer> where X is one of A, B, C, or D."
        else:
            # Direct prompting
            return f"{question}{options_text}\n\nPlease provide your final answer in the format <answer>X</answer> where X is one of A, B, C, or D."
    
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
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
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
        
        answer_counts = Counter(answers)
        total_count = len(answers)
        
        probabilities = {}
        unique_options = list(answer_counts.keys())
        
        for answer, count in answer_counts.items():
            probabilities[answer] = count / total_count
        
        logger.info(f"Generated {len(unique_options)} unique options from {total_count} samples: {answer_counts}")
        
        return probabilities, unique_options
    
    def _compute_option_probabilities(self, response: str, options: List[str]) -> Dict[str, float]:
        """Compute softmax probabilities for each option."""
        start_pos, end_pos = self._find_answer_positions(response)
        
        if start_pos == -1:
            uniform_prob = 1.0 / len(options)
            return {option: uniform_prob for option in options}
        
        probabilities = {}
        logits = []
        
        for option in options:
            modified_response = response[:start_pos] + option + response[end_pos:]
            
            inputs = self.tokenizer.encode(modified_response, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                option_tokens = self.tokenizer.encode(option, add_special_tokens=False)
                if not option_tokens:
                    logits.append(-float('inf'))
                    continue
                    
                option_token_id = option_tokens[0]
                input_ids = inputs[0].cpu().numpy()
                
                option_pos = -1
                for i in range(len(input_ids) - len(option_tokens) + 1):
                    if np.array_equal(input_ids[i:i+len(option_tokens)], option_tokens):
                        option_pos = i
                        break
                
                if option_pos > 0:
                    logit = outputs.logits[0, option_pos-1, option_token_id].item()
                else:
                    logit = outputs.logits[0, -1, option_token_id].item()
                
                logits.append(logit)
        
        logits_tensor = torch.tensor(logits)
        probs_tensor = F.softmax(logits_tensor, dim=0)
        
        for i, option in enumerate(options):
            probabilities[option] = probs_tensor[i].item()
        
        return probabilities
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample through the Fusion LLM system."""
        # Generate diverse queries for the question
        question = sample.get('question', '')
        diverse_queries = sample.get('diverse_queries', [])
        if not diverse_queries:
            # Fallback: tạo mới nếu không có
            diverse_queries = self.generate_diverse_queries(sample.get('question', ''))
        # Generate prompt
        prompt = self._generate_prompt(sample)
        
        # Extract options from the sample
        options = sample.get('options', [])
        
        if not options:
            # Case 1: Empty options - generate multiple responses
            logger.info(f"Empty options detected. Generating {self.num_samples} responses...")
            
            answers = self._generate_multiple_responses(prompt, self.num_samples)
            option_probabilities, generated_options = self._compute_probabilities_from_samples(answers)
            
            if option_probabilities:
                predicted_answer = max(option_probabilities.items(), key=lambda x: x[1])[0]
            else:
                predicted_answer = "Unknown"
            
            final_response = self._generate_response(prompt, temperature=0.1)
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': final_response,
                'predicted_answer': predicted_answer,
                'option_probabilities': option_probabilities,
                'num_samples_generated': len(answers),
                'prompt_used': prompt,
                'technique': sample.get('technique', 'fusion'),
                'method': 'frequency_based',
                'diverse_queries': diverse_queries,
                'system_type': 'fusion_llm'
            }
        else:
            # Case 2: Options provided
            logger.info(f"Using provided options: {options}")
            
            response = self._generate_response(prompt, temperature=0.1)
            option_probabilities = self._compute_option_probabilities(response, options)
            predicted_answer = self._extract_answer(response)
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': response,
                'predicted_answer': predicted_answer,
                'option_probabilities': option_probabilities,
                'provided_options': options,
                'prompt_used': prompt,
                'technique': sample.get('technique', 'fusion'),
                'method': 'logit_based',
                'diverse_queries': diverse_queries,
                'system_type': 'fusion_llm'
            }
