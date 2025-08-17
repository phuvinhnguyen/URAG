from abc import ABC, abstractmethod
from typing import Dict, Any, List
from tqdm.auto import tqdm
from loguru import logger

class AbstractRAGSystem(ABC):
    """
    Abstract base class for RAG systems.
    
    This allows for flexible integration of different system architectures:
    - Simple LLM systems
    - Traditional RAG systems
    - Complex multi-step reasoning systems
    - Hybrid approaches
    """
    
    @abstractmethod
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
        for sample in tqdm(samples, desc="Processing samples"):
            try:
                result = self.process_sample(sample)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                # Add error result
                error_result = {
                    'id': sample.get('id', 'unknown'),
                    'generated_response': f"Error: {str(e)}",
                    'predicted_answer': "Error",
                    'option_probabilities': {},
                    'error': str(e)
                }
                results.append(error_result)
        
        return results
