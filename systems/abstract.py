from abc import ABC, abstractmethod
from typing import Dict, Any, List
from loguru import logger
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F

class EndSequenceCriteria(StoppingCriteria):
    def __init__(self, end_ids):
        super().__init__()
        self.end_ids = end_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if len(input_ids[0]) < len(self.end_ids) + 1:
            return False
        result = torch.equal(input_ids[0][-len(self.end_ids)-1:-1], torch.tensor(self.end_ids, device=input_ids.device))
        return result

class AbstractRAGSystem(ABC):
    """
    Abstract base class for RAG systems.
    
    This allows for flexible integration of different system architectures:
    - Simple LLM systems
    - Traditional RAG systems
    - Complex multi-step reasoning systems
    - Hybrid approaches
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
                output_scores=True
            )

        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        last_token_probs = F.softmax(outputs.scores[-1][0], dim=-1)
        option_tokens = [self.tokenizer.encode(option, add_special_tokens=False)[0] for option in options]
        last_token_probs = last_token_probs[option_tokens] + 1e-10
        last_token_probs = last_token_probs / last_token_probs.sum()
        conformal_probabilities = {option: last_token_probs[i].item() for i, option in enumerate(options)}
        return generated_text, conformal_probabilities

    def _generate_response_with_probabilities(self, prompt: str, options: List[str]):
        if getattr(self, 'method', 'normal') == 'aware':
            _, probs = self._generate_response_with_probabilities_normal(prompt, options)
            confidence_text = ' Knowing that the agent\'s confidence for each answer is ' + ', '.join([f"{k}: {v:.4f}" for k, v in probs.items()])
            return self._generate_response_with_probabilities_normal(prompt + confidence_text, options)
        else:
            return self._generate_response_with_probabilities_normal(prompt, options)

