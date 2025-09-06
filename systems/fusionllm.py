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
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", num_samples: int = 20, num_queries: int = 3, technique: str = "direct"):
        """Initialize the Fusion LLM system."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.technique = technique
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.num_samples = num_samples
        self.num_queries = num_queries  # Number of diverse queries to generate
        
        # Try to load the requested model, with fallback
        self.model_name = self._load_model_with_fallback(model_name)
        
        logger.info(f"Successfully loaded Fusion model: {self.model_name}")
    
    def _load_model_with_fallback(self, preferred_model: str) -> str:
        """Load model with automatic fallback if preferred model fails."""
        # List of open-source models to try in order of preference
        fallback_models = [
            preferred_model,
            "mistralai/Mistral-7B-Instruct-v0.1",
            "tiiuae/falcon-7b-instruct",
            "mosaicml/mpt-7b-chat",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small", 
            "gpt2-medium",
            "gpt2",
            "distilgpt2"
        ]
        
        for model_name in fallback_models:
            try:
                logger.info(f"Attempting to load model: {model_name}")
                
                # Standard model loading approach for all models
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    padding_side="left"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None,
                    trust_remote_code=True
                )
                
                if self.device.type != "cuda" or "device_map" not in locals():
                    self.model.to(self.device)
                
                self.model.eval()
                
                # Add pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                if model_name != preferred_model:
                    logger.warning(f"Preferred model '{preferred_model}' failed, using fallback: '{model_name}'")
                
                return model_name
                
            except Exception as e:
                logger.warning(f"Failed to load model '{model_name}': {str(e)}")
                continue
        
        raise RuntimeError(f"Failed to load any compatible model. Last attempted: {fallback_models}")
    
    def get_batch_size(self) -> int:
        """Return batch size of 1 for this implementation."""
        return 1
    
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
    
    def generate_diverse_queries(self, question: str) -> List[str]:
        """Generate multiple diverse queries from the original question."""
        queries = [question]  # Include original question
        
        # Generate diverse query prompts using unified format
        query_templates = [
            ("You are a helpful assistant that rephrases questions in different ways while maintaining the same meaning.", 
             f"Rephrase this question in a different way while keeping the same meaning:\n\nOriginal question: {question}\n\nProvide only the rephrased question:"),
            ("You are a helpful assistant that creates alternative formulations of questions.", 
             f"What is another way to ask about the same topic as this question:\n\nOriginal question: {question}\n\nProvide only the alternative question:"),
            ("You are a helpful assistant that generates related questions.", 
             f"Generate a related question that might help answer this question:\n\nOriginal question: {question}\n\nProvide only the related question:")
        ]
        
        for i, (system_msg, user_msg) in enumerate(query_templates[:self.num_queries-1]):  # -1 because we already have original
            try:
                prompt = self._create_unified_prompt(system_msg, user_msg)
                generated_query = self._generate_response(prompt, max_length=100, temperature=0.8)
                
                # Clean up the generated query
                generated_query = generated_query.split('\n')[0].strip()
                # Remove any leading/trailing quotes or punctuation
                generated_query = generated_query.strip('"\'`.,!?')
                
                if generated_query and generated_query not in queries and len(generated_query) > 10:
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
        elif technique == 'rag' or technique == 'fusion':
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
        """Process a single sample through the Fusion LLM system."""
        # Generate diverse queries for the question
        question = sample.get('question', '')
        diverse_queries = sample.get('diverse_queries', [])
        if not diverse_queries:
            # Fallback: generate new queries if not provided
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
                'technique': self.technique,
                'method': 'frequency_based',
                'diverse_queries': diverse_queries,
                'system_type': 'fusion_llm'
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
                'diverse_queries': diverse_queries,
                'system_type': 'fusion_llm'
            }
