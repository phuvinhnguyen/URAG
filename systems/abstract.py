from abc import ABC, abstractmethod
from ast import Tuple
from typing import Dict, Any, List, Union
from loguru import logger
import torch
from transformers import StoppingCriteria, AutoTokenizer
from vllm import LLM, SamplingParams

class AbstractRAGSystem(ABC):
    """
    Abstract base class for RAG systems.

    This allows for flexible integration of different system architectures:
    - Simple LLM systems
    - Traditional RAG systems
    - Complex multi-step reasoning systems
    - Hybrid approaches

    Note: Implementations can use either approach:

    For high-performance vllm-based systems:
    - self.vllm_model: vllm.LLM instance for high-performance inference
    - self.model_name: str, the model name/path for tokenizer initialization
    - self.temperature: float, sampling temperature
    - self.max_new_tokens: int, maximum tokens to generate

    For traditional HuggingFace-based systems (backward compatibility):
    - self.model: HuggingFace model instance
    - self.tokenizer: HuggingFace tokenizer instance
    - self.device: device for model inference
    - self.temperature: float, sampling temperature
    - self.max_new_tokens: int, maximum tokens to generate
    """
    method = 'normal'
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample and return results.
        
        Args:
            sample: Input sample dictionary with question, options, etc.
            
        Returns:
            Dictionary containing at minimum:
            - 'generated_response': The system's raw response
            - 'predicted_answer': Extracted answer (should follow <answer>X</answer> format)
            - 'option_probabilities': Dict mapping options to probabilities
            
            Optionally can include:
            - 'reasoning': Intermediate reasoning steps
            - 'retrieved_docs': Retrieved documents for RAG
            - 'confidence': System confidence score
            - 'processing_time': Time taken to process
        """
        pass
    
    @abstractmethod
    def get_batch_size(self) -> int:
        """Return the preferred batch size for this system."""
        pass
    
    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of samples. Default implementation processes one by one.
        Override this for systems that can benefit from true batch processing.
        
        Args:
            samples: List of input samples
            
        Returns:
            List of processed results
        """
        results = []
        for sample in samples:
            try:
                result = self.process_sample(sample)
                results.append(result)
            except Exception as e:
                logger.exception(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
        
        return results
    
    def _generate_response_with_probabilities_normal(self, prompt: str, options: List[str]):
        # Check if using vllm or traditional HuggingFace approach
        if hasattr(self, 'vllm_model'):
            return self._generate_response_with_probabilities_vllm(prompt, options)
        else:
            return self._generate_response_with_probabilities_hf(prompt, options)

    def _generate_response_with_probabilities_vllm(self, prompt: str, options: List[str]):
        """Generate response using vllm for improved performance."""
        # Create sampling parameters for vllm
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            stop=["Answer|"],
            logprobs=len(options),  # Get logprobs for calculating option probabilities
            prompt_logprobs=1  # Get prompt logprobs
        )

        # Generate response using vllm
        outputs = self.vllm_model.generate([prompt], sampling_params)
        output = outputs[0]

        generated_text = prompt + output.outputs[0].text

        # Extract token probabilities for options
        if output.outputs[0].logprobs:
            # Get the last generated token's logprobs
            last_token_logprobs = output.outputs[0].logprobs[-1] if output.outputs[0].logprobs else {}

            # Get tokenizer for encoding options
            if not hasattr(self, '_tokenizer_for_vllm'):
                self._tokenizer_for_vllm = AutoTokenizer.from_pretrained(self.model_name)

            # Get option tokens
            option_tokens = []
            for option in options:
                tokens = self._tokenizer_for_vllm.encode(option, add_special_tokens=False)
                if tokens:
                    option_tokens.append(tokens[0])
                else:
                    # Fallback: use the token ID from vllm's vocab
                    option_tokens.append(self._tokenizer_for_vllm.vocab.get(option, self._tokenizer_for_vllm.unk_token_id))

            # Extract probabilities for option tokens
            option_logprobs = []
            for token_id in option_tokens:
                if token_id in last_token_logprobs:
                    option_logprobs.append(last_token_logprobs[token_id].logprob)
                else:
                    option_logprobs.append(float('-inf'))  # Very low probability

            # Convert logprobs to probabilities and normalize
            option_probs = torch.softmax(torch.tensor(option_logprobs), dim=-1)
            conformal_probabilities = {option: option_probs[i].item() for i, option in enumerate(options)}
        else:
            # Fallback: uniform probabilities if logprobs not available
            uniform_prob = 1.0 / len(options)
            conformal_probabilities = {option: uniform_prob for option in options}

        return generated_text, conformal_probabilities

    def _generate_response_with_probabilities_hf(self, prompt: Union[str, List[str]], options: Union[List[str], List[List[str]]]):
        """Generate response using traditional HuggingFace transformers with proper batch processing."""
        from transformers import StoppingCriteriaList, StoppingCriteria
        import torch.nn.functional as F
        
        class EndSequenceCriteria(StoppingCriteria):
            def __init__(self, end_ids):
                super().__init__()
                self.end_ids = end_ids
                
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                if len(input_ids[0]) < len(self.end_ids) + 1:
                    return False
                result = torch.equal(
                    input_ids[0][-len(self.end_ids)-1:-1], 
                    torch.tensor(self.end_ids, device=input_ids.device)
                )
                return result
        
        # Determine if single or batch processing
        is_batch = isinstance(prompt, list)
        
        if not is_batch:
            # Single sample processing
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
                    stopping_criteria=stopping,
                    return_dict_in_generate=True,
                    output_logits=True
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            conformal_probabilities = self._extract_answer_probabilities(generated_text, options)
            
            return generated_text, conformal_probabilities
        
        else:
            # Batch processing
            batch_size = len(prompt)
            
            # Handle options: if single list, broadcast to all samples
            if isinstance(options[0], str):
                # options is List[str], broadcast to all samples
                batch_options = [options] * batch_size
            else:
                # options is List[List[str]], each sample has its own options
                batch_options = options
            
            # Tokenize batch
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            start_ids = self.tokenizer.encode("Answer|", add_special_tokens=False)
            stopping = StoppingCriteriaList([EndSequenceCriteria(start_ids)])
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping,
                    return_dict_in_generate=True,
                    output_logits=True
                )
            
            # Process each sample in batch
            generated_texts = []
            conformal_probabilities_list = []
            
            for i in range(batch_size):
                # Decode text for each sample
                generated_text = self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
                generated_texts.append(generated_text)

                conformal_probabilities_list.append(self._extract_answer_probabilities(generated_text, batch_options[i]))
            
            return generated_texts, conformal_probabilities_list

    def _extract_answer_probabilities(self, generated_response: str, options: List[str]) -> Dict[str, float]:
        """
        Extract probabilities for answer options from a generated response.
        
        This function finds where "Answer|" appears in the generated response,
        then extracts logits for all possible option letters at that position.
        
        Args:
            generated_response: The full generated response string
            options: List of option letters (e.g., ["A", "B", "C", "D"])
        
        Returns:
            Dictionary mapping option letters to probabilities
        """
        # Find the position where "Answer|" appears (get the last occurrence)
        answer_prefix = "Answer|"
        answer_pos = generated_response.rfind(answer_prefix)
        
        if answer_pos == -1:
            # If "Answer|" not found, use the last token position
            inputs = self.tokenizer(generated_response, return_tensors="pt", truncation=True)
            for k in inputs:
                if torch.is_tensor(inputs[k]):
                    inputs[k] = inputs[k].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze(0)  # Last token logits
        else:
            # Get text up to and including "Answer|" (without the actual answer letter)
            prefix_text = generated_response[:answer_pos + len(answer_prefix)]
            
            # Tokenize the prefix
            prefix_inputs = self.tokenizer(prefix_text, return_tensors="pt", truncation=True)
            for k in prefix_inputs:
                if torch.is_tensor(prefix_inputs[k]):
                    prefix_inputs[k] = prefix_inputs[k].to(self.device)
            
            # Forward pass to get logits at the position right after "Answer|"
            with torch.no_grad():
                outputs = self.model(**prefix_inputs)
            logits = outputs.logits[:, -1, :].squeeze(0)  # Logits at position after "Answer|"
        
        # Get token IDs for each option letter
        # For Llama tokenizer, single letters are typically tokenized as single tokens
        option_logits_list = []
        
        for opt in options:
            # Encode just the letter to get its token ID
            token_ids = self.tokenizer.encode(opt, add_special_tokens=False)
            if len(token_ids) > 0:
                token_id = token_ids[-1]  # Use the last token (in case of multi-token encoding)
                if token_id < logits.shape[0]:  # Check bounds
                    option_logits_list.append((opt, logits[token_id].item()))
                else:
                    option_logits_list.append((opt, -100.0))  # Out of bounds, use small logit
            else:
                # Fallback: try with space
                token_ids = self.tokenizer.encode(f" {opt}", add_special_tokens=False)
                if len(token_ids) > 0:
                    token_id = token_ids[-1]
                    if token_id < logits.shape[0]:
                        option_logits_list.append((opt, logits[token_id].item()))
                    else:
                        option_logits_list.append((opt, -100.0))
                else:
                    option_logits_list.append((opt, -100.0))
        
        # Extract logit values
        logit_values = torch.tensor([logit for _, logit in option_logits_list], dtype=torch.float32)
        
        # Compute probabilities using softmax
        probs = torch.nn.functional.softmax(logit_values, dim=0)
        
        # Create probability dictionary
        option_probs = {}
        for i, (opt, _) in enumerate(option_logits_list):
            option_probs[opt] = float(probs[i].item())
        
        return option_probs

    def _prompt_injection(self, prompt: str, injection_text: str) -> str:
        parts = prompt.split(self.tokenizer.eos_token)
        parts = parts[:-1] + [injection_text, parts[-1]]
        return self.tokenizer.eos_token.join(parts)

    def _generate_response_with_probabilities(self, prompt: Union[str, List[str]], options: Union[List[str]]):
        if getattr(self, 'method', 'normal') == 'aware':
            if isinstance(prompt, str):
                _, probs = self._generate_response_with_probabilities_normal(prompt, options)
                confidence_text = ' Your previous confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in probs.items()]) + '. My answer to follow the format is:\n\n'
                return self._generate_response_with_probabilities_normal(self._prompt_injection(prompt, confidence_text), options)
            else:
                _, batch_probs = self._generate_response_with_probabilities_normal(prompt, options)
                confidence_text = [' Your previous confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in probs.items()]) + '. My answer to follow the format is:\n\n' for probs in batch_probs]
                return self._generate_response_with_probabilities_normal([self._prompt_injection(x, y) for x, y in zip(prompt, confidence_text)], options)
        elif getattr(self, 'method', 'normal') == 'attack':
            _, batch_probs = self._generate_response_with_probabilities_normal(prompt, options)
            if isinstance(prompt, str):
                confidence_text = ' Your previous confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in zip(batch_probs.keys(), [list(batch_probs.values())[-1]] + list(batch_probs.values())[:-1])]) + '. My answer to follow the format is:\n\n'
                prompt = self._prompt_injection(prompt, confidence_text)
            else:
                confidence_text = [' Your previous confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in zip(probs.keys(), [list(probs.values())[-1]] + list(probs.values())[:-1])]) + '. My answer to follow the format is:\n\n' for probs in batch_probs]
                prompt = [self._prompt_injection(x, y) for x, y in zip(prompt, confidence_text)]
            return self._generate_response_with_probabilities_normal(prompt, options)
        elif getattr(self, 'method', 'normal') == 'defense':
            if isinstance(prompt, str):
                confidence_text = ' Your previous confidence for each answer is ' + ', '.join([f"{k}: {1/len(options)}" for k in options]) + '. My reason and answer to follow the format is:\n\n'
                return self._generate_response_with_probabilities_normal(self._prompt_injection(prompt, confidence_text), options)
            else:
                confidence_text = [' Your previous confidence for each answer is ' + ', '.join([f"{k}: {1/len(option)}" for k in option]) + '. My reason and answer to follow the format is:\n\n' for option in options]
                return self._generate_response_with_probabilities_normal([self._prompt_injection(x, y) for x, y in zip(prompt, confidence_text)], options)
        else:
            return self._generate_response_with_probabilities_normal(prompt, options)


    # def _generate_response_with_probabilities(self, prompt: Union[str, List[str]], options: Union[List[str]]):
    #     if getattr(self, 'method', 'normal') == 'aware':
    #         if isinstance(prompt, str):
    #             _, probs = self._generate_response_with_probabilities_normal(prompt, options)
    #             confidence_text = ' My confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in probs.items()]) + '. My answer to follow the format is:\n\n'
    #             return self._generate_response_with_probabilities_normal(prompt + confidence_text, options)
    #         else:
    #             _, batch_probs = self._generate_response_with_probabilities_normal(prompt, options)
    #             confidence_text = [' My confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in probs.items()]) + '. My answer to follow the format is:\n\n' for probs in batch_probs]
    #             return self._generate_response_with_probabilities_normal([x+y for x, y in zip(prompt, confidence_text)], options)
    #     elif getattr(self, 'method', 'normal') == 'attack':
    #         _, batch_probs = self._generate_response_with_probabilities_normal(prompt, options)
    #         if isinstance(prompt, str):
    #             confidence_text = ' My confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in zip(batch_probs.keys(), [list(batch_probs.values())[-1]] + list(batch_probs.values())[:-1])]) + '. My answer to follow the format is:\n\n'
    #             prompt = prompt + confidence_text
    #         else:
    #             confidence_text = [' My confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in zip(probs.keys(), [list(probs.values())[-1]] + list(probs.values())[:-1])]) + '. My answer to follow the format is:\n\n' for probs in batch_probs]
    #             prompt = [x+y for x, y in zip(prompt, confidence_text)]
    #         return self._generate_response_with_probabilities_normal(prompt, options)
    #     elif getattr(self, 'method', 'normal') == 'defense':
    #         if isinstance(prompt, str):
    #             confidence_text = ' My confidence for each answer is ' + ', '.join([f"{k}: {1/len(options)}" for k in options]) + '. My reason and answer to follow the format is:\n\n'
    #             return self._generate_response_with_probabilities_normal(prompt + confidence_text, options)
    #         else:
    #             confidence_text = [' My confidence for each answer is ' + ', '.join([f"{k}: {1/len(option)}" for k in option]) + '. My reason and answer to follow the format is:\n\n' for option in options]
    #             return self._generate_response_with_probabilities_normal([x+y for x, y in zip(prompt, confidence_text)], options)
    #     elif getattr(self, 'method', 'normal') == 'userconf':
    #         _, batch_probs = self._generate_response_with_probabilities_normal(prompt, options)
    #         if isinstance(prompt, str):
    #             confidence_text = ' The user\'s confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in zip(batch_probs.keys(), [list(batch_probs.values())[-1]] + list(batch_probs.values())[:-1])]) + '. My answer to follow the format is:\n\n'
    #             prompt = prompt + confidence_text
    #         else:
    #             confidence_text = [' The user\'s confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in zip(probs.keys(), [list(probs.values())[-1]] + list(probs.values())[:-1])]) + '. My answer to follow the format is:\n\n' for probs in batch_probs]
    #             prompt = [x+y for x, y in zip(prompt, confidence_text)]
    #         return self._generate_response_with_probabilities_normal(prompt, options)
    #     else:
    #         return self._generate_response_with_probabilities_normal(prompt, options)

