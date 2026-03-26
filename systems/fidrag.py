from systems.abstract import AbstractRAGSystem
from systems.fidllm import FiDLLMSystem
from typing import Dict, Any, List
import torch
from loguru import logger
from utils.clean import clean_web_content  # pyright: ignore[reportMissingImports]
from utils.storage import get_storage  # pyright: ignore[reportMissingImports]
from utils.ramdb import ChunkSearcher

# Only need transformers for FiDGenerator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Removed FiDGenerator class - not needed anymore since we use ChunkSearcher directly


class FiDRAGSystem(AbstractRAGSystem):
    """
    FiD RAG system that uses Fusion-in-Decoder approach for retrieval and generation.
    
    This system implements the FiD (Fusion-in-Decoder) technique:
    1. Retrieve relevant documents using BM25 or dense retrieval
    2. Create separate contexts for each document with the question
    3. Use FiD model to fuse information from all contexts in the decoder
    4. Generate final answer using the fused representation
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base", fid_model_name: str = "Intel/fid_flan_t5_base_nq", device: str = "auto", num_samples: int = 20, retrieved_docs: int = 10, embedding_model: str = "all-MiniLM-L6-v2", **kwargs):
        self.retrieved_docs = retrieved_docs
        self.llm_system = FiDLLMSystem(model_name, fid_model_name, device, num_samples=num_samples, technique='rag', **kwargs)
        self.embedding_model = embedding_model
        self.fid_model_name = fid_model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    
    def _extract_answer_from_response(self, response: str) -> str:
        """Extract answer from structured response format - returns only A, B, C, D, or E."""
        import re
        
        # Look for Answer|X pattern
        match = re.search(r'Answer\|([A-E])', response)
        if match:
            return match.group(1)
        
        # Fallback: look for single letter answers at the end or beginning
        response_clean = response.strip()
        if len(response_clean) == 1 and response_clean in 'ABCDE':
            return response_clean
        
        # Look for pattern like "A.", "B)", etc.
        match = re.search(r'\b([A-E])[.)]\s*$', response)
        if match:
            return match.group(1)
        
        # Last resort: return first letter found in ABCDE
        for char in response:
            if char in 'ABCDE':
                return char
        
        return 'A'  # Default fallback
    
    def get_batch_size(self) -> int: return 80
    
    # Removed _create_haystack_pipeline method - not needed anymore
    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of samples using FiD approach - simplified like HyDERAG."""
        results = []
        
        # Setup database - unified approach like HyDERAG
        sample = samples[0]
        if sample.get('search_results', []) != [] and \
            sample['search_results'][0].get('persistent_storage', None):
            # Use persistent storage
            if not hasattr(self, 'database'):
                self.database = ChunkSearcher(embedding_model=self.embedding_model)
                self.database.set_documents([get_storage(sample['search_results'][0]['persistent_storage'])])
            database = self.database
        else:
            # Use search results from HTML content - same as HyDERAG
            documents = [[doc['page_snippet'] + "\n\n" + clean_web_content(doc.get('page_result', ''))
                        for doc in _sample.get('search_results', [])] for _sample in samples]
            database = ChunkSearcher(embedding_model=self.embedding_model)
            database.set_documents(documents)
        
        # Process each sample - unified approach
        for _id, sample in enumerate(samples):
            try:
                question = sample.get('question', '')
                query_time = sample.get('query_time', 'March 1, 2025')
                
                # Retrieve documents using the database
                retrieved_docs = database.search(question, k=self.retrieved_docs)
                
                # Create augmented sample with context
                augmented_sample = sample.copy()
                if retrieved_docs: augmented_sample['context'] = ("\n- " + "\n- ".join(retrieved_docs))[:4000] + '\nQuery Time: ' + query_time
                
                # Use FiD processing
                result = self.llm_system.process_sample_fid(augmented_sample)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                results.append({
                    'id': sample.get('id', 'unknown'),
                    'generated_response': "",
                    'predicted_answer': "",
                    'conformal_probabilities': {},
                    'technique': 'fid_error',
                    'error': str(e)
                })
        
        return results
