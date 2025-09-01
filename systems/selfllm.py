from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import re
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from collections import Counter

class SelfLLMSystem(AbstractRAGSystem):
    """
    Self-RAG LLM system that implements self-reflection for retrieval and generation.
    
    This system uses special reflection tokens to evaluate:
    - Whether retrieval is needed
    - Whether retrieved documents are relevant
    - Whether the generated answer is supported by the context
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", num_samples: int = 20, technique: str = "direct"):
        """Initialize the Self-RAG LLM system."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.num_samples = num_samples
        self.technique = technique
        logger.info(f"Loading Self-RAG model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Define self-reflection tokens
        self.reflection_tokens = {
            'retrieve': ['<retrieve>Yes</retrieve>', '<retrieve>No</retrieve>'],
            'relevant': ['<relevant>Yes</relevant>', '<relevant>No</relevant>'],
            'support': ['<support>Fully supported</support>', '<support>Partially supported</support>', '<support>Not supported</support>'],
            'utility': ['<utility>Helpful</utility>', '<utility>Not helpful</utility>']
        }
    
    def get_batch_size(self) -> int:
        """Return batch size of 1 for this implementation."""
        return 1
    
    def evaluate_retrieval_need(self, question: str) -> bool:
        """Evaluate whether retrieval is needed for the given question."""
        prompt = f"""Question: {question}

Do you need to retrieve additional information to answer this question accurately?
Please respond with <retrieve>Yes</retrieve> or <retrieve>No</retrieve>.

Response: """
        
        response = self._generate_response(prompt, max_length=50, temperature=0.1)
        return '<retrieve>Yes</retrieve>' in response
    
    def evaluate_relevance(self, question: str, context: str) -> bool:
        """Evaluate whether the retrieved context is relevant to the question."""
        prompt = f"""Question: {question}

Retrieved Context: {context}

Is this context relevant and helpful for answering the question?
Please respond with <relevant>Yes</relevant> or <relevant>No</relevant>.

Response: """
        
        response = self._generate_response(prompt, max_length=50, temperature=0.1)
        return '<relevant>Yes</relevant>' in response
    
    def evaluate_support(self, question: str, context: str, answer: str) -> str:
        """Evaluate whether the answer is supported by the context."""
        prompt = f"""Question: {question}

Context: {context}

Answer: {answer}

Is this answer fully supported by the provided context?
Please respond with one of:
- <support>Fully supported</support>
- <support>Partially supported</support>
- <support>Not supported</support>

Response: """
        
        response = self._generate_response(prompt, max_length=50, temperature=0.1)
        
        if '<support>Fully supported</support>' in response:
            return 'Fully supported'
        elif '<support>Partially supported</support>' in response:
            return 'Partially supported'
        else:
            return 'Not supported'
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt based on sample technique."""
        question = sample.get('question', '')
        technique = self.technique
        
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
        elif technique == 'rag' or technique == 'self':
            # For Self-RAG, use context if available with reflection
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
        """Process a single sample through the Self-RAG LLM system."""
        question = sample.get('question', '')
        context = sample.get('search_results', sample.get('context', ''))
        
        # Self-reflection evaluations
        retrieval_needed = self.evaluate_retrieval_need(question) if not context else True
        relevance_score = self.evaluate_relevance(question, context) if context else False
        
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
            
            # Evaluate support for the generated answer
            support_score = self.evaluate_support(question, context, final_response) if context else 'Not evaluated'
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': final_response,
                'predicted_answer': predicted_answer,
                'option_probabilities': option_probabilities,
                'num_samples_generated': len(answers),
                'prompt_used': prompt,
                'technique': self.technique,
                'method': 'frequency_based',
                'system_type': 'self_llm',
                'retrieval_needed': retrieval_needed,
                'relevance_score': relevance_score,
                'support_score': support_score,
                'self_reflection': {
                    'retrieval_needed': retrieval_needed,
                    'context_relevant': relevance_score,
                    'answer_supported': support_score
                }
            }
        else:
            # Case 2: Options provided
            logger.info(f"Using provided options: {options}")
            
            response = self._generate_response(prompt, temperature=0.1)
            option_probabilities = self._compute_option_probabilities(response, options)
            predicted_answer = max(option_probabilities.items(), key=lambda x: x[1])[0]
            
            # Evaluate support for the generated answer
            support_score = self.evaluate_support(question, context, response) if context else 'Not evaluated'
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': response,
                'predicted_answer': predicted_answer,
                'option_probabilities': option_probabilities,
                'provided_options': options,
                'prompt_used': prompt,
                'technique': self.technique,
                'method': 'logit_based',
                'system_type': 'self_llm',
                'retrieval_needed': retrieval_needed,
                'relevance_score': relevance_score,
                'support_score': support_score,
                'self_reflection': {
                    'retrieval_needed': retrieval_needed,
                    'context_relevant': relevance_score,
                    'answer_supported': support_score
                }
            }
