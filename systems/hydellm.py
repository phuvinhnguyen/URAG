from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import re
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from collections import Counter

class HyDELLMSystem(AbstractRAGSystem):
    """
    HyDE LLM system that generates hypothetical documents for better retrieval.
    
    This system first generates a hypothetical document that would answer the query,
    then uses that document for retrieval instead of the original query.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", num_samples: int = 20, technique: str = "direct"):
        """Initialize the HyDE LLM system."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.num_samples = num_samples
        self.technique = technique
        logger.info(f"Loading HyDE model {model_name} on {self.device}")
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
    
    def generate_hypothetical_document(self, question: str) -> str:
        """Generate a hypothetical document that would answer the question."""
        # Unified prompt format that works well across different models
        prompt = self._create_unified_prompt(
            system_message="You are a helpful assistant that writes comprehensive and informative passages to answer questions.",
            user_message=f"Write a detailed, factual passage that would answer the following question:\n\nQuestion: {question}\n\nProvide a comprehensive answer with relevant facts and information."
        )
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 200,  # Generate longer hypothetical document
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        hypothetical_doc = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return hypothetical_doc.strip()
    
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
        
        # Get context if available
        context = sample.get('search_results', sample.get('context', ''))
        
        # Generate unified prompts based on technique
        if technique == 'cot':
            system_message = "You are a helpful assistant that answers multiple choice questions with step-by-step reasoning. Think through the problem carefully and provide your final answer in the format <answer>X</answer>."
            user_message = f"Let's think step by step.\n\n{question}{options_text}\n\nPlease provide your reasoning and then give your final answer in the format <answer>X</answer> where X is one of A, B, C, or D."
        elif technique == 'rag' or technique == 'hyde':
            if context:
                system_message = "You are a helpful assistant that answers multiple choice questions using the provided context information. Use the context to inform your answer and provide the final answer in the format <answer>X</answer>."
                user_message = f"Context information: {context}\n\nQuestion: {question}{options_text}\n\nPlease provide your final answer in the format <answer>X</answer> where X is one of A, B, C, or D."
            else:
                system_message = "You are a helpful assistant that answers multiple choice questions. Provide your final answer in the format <answer>X</answer>."
                user_message = f"{question}{options_text}\n\nPlease provide your final answer in the format <answer>X</answer> where X is one of A, B, C, or D."
        else:
            # Direct prompting
            system_message = "You are a helpful assistant that answers multiple choice questions. Provide your final answer in the format <answer>X</answer>."
            user_message = f"{question}{options_text}\n\nPlease provide your final answer in the format <answer>X</answer> where X is one of A, B, C, or D."
        
        return self._create_unified_prompt(system_message, user_message)
    
    def _generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate response from the LLM."""
        # Adaptive input length based on model capabilities
        model_lower = self.model_name.lower()
        if any(x in model_lower for x in ["llama", "mistral", "falcon"]):
            max_input_length = 2048
        else:
            max_input_length = 512
            
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs = inputs.to(self.device)
        
        # Base generation configuration
        generation_config = {
            "max_length": inputs.shape[1] + max_length,
            "num_return_sequences": 1,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1
        }
        
        # Model-specific optimizations
        if any(x in model_lower for x in ["llama", "mistral", "falcon"]):
            generation_config.update({
                "top_p": 0.9,
                "top_k": 40
            })
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, **generation_config)
        
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
        """Extract answer from response with multiple patterns."""
        # Try primary pattern first
        start_pos, end_pos = self._find_answer_positions(response)
        if start_pos != -1:
            return response[start_pos:end_pos].strip()
        
        # Fallback patterns for better parsing
        patterns = [
            r'answer.*?([ABCD])',
            r'choice.*?([ABCD])', 
            r'option.*?([ABCD])',
            r'correct.*?([ABCD])',
            r'([ABCD])\.?\s*(?:is|are|would|should)',
            r'(?:choose|select).*?([ABCD])',
            r'\b([ABCD])\b(?=\s*[\.\)\-\:])'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                if answer in ['A', 'B', 'C', 'D']:
                    logger.debug(f"Extracted answer '{answer}' using pattern: {pattern}")
                    return answer
        
        logger.warning(f"Could not extract answer from response: {response[:100]}...")
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
        """Process a single sample through the HyDE LLM system."""
        # Check if hypothetical document is already provided (from HyDERAGSystem)
        question = sample.get('question', '')
        hypothetical_doc = sample.get('hypothetical_document', '')
        
        # Only generate hypothetical document if not provided (standalone mode)
        if not hypothetical_doc:
            hypothetical_doc = self.generate_hypothetical_document(question)
            logger.info("Generated hypothetical document in standalone mode")
        else:
            logger.info("Using provided hypothetical document from HyDERAGSystem")
        
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
                'technique': self.technique,
                'method': 'frequency_based',
                'hypothetical_document': hypothetical_doc,
                'system_type': 'hyde_llm',
                'hypothetical_doc_regenerated': 'hypothetical_document' not in sample
            }
        else:
            # Case 2: Options provided
            logger.info(f"Using provided options: {options}")
            
            response = self._generate_response(prompt, temperature=0.1)
            option_probabilities = self._compute_option_probabilities(response, options)
            predicted_answer = max(option_probabilities.items(), key=lambda x: x[1])[0]
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': response,
                'predicted_answer': predicted_answer,
                'option_probabilities': option_probabilities,
                'provided_options': options,
                'prompt_used': prompt,
                'technique': self.technique,
                'method': 'logit_based',
                'hypothetical_document': hypothetical_doc,
                'system_type': 'hyde_llm',
                'hypothetical_doc_regenerated': 'hypothetical_document' not in sample
            }
