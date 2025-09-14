from systems.abstract import AbstractRAGSystem
from systems.fusionllm import FusionLLMSystem
from typing import Dict, Any, List, Tuple
from loguru import logger
import numpy as np
from collections import defaultdict
from utils.clean import clean_web_content  # pyright: ignore[reportMissingImports]
from utils.ramdb import ChunkSearcher
from utils.storage import get_storage  # pyright: ignore[reportMissingImports]
import hashlib


def hash_string(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

class FusionRAGSystem(AbstractRAGSystem):
    """
    Fusion RAG system that uses multiple diverse queries and Reciprocal Rank Fusion (RRF).
    
    This system implements the Fusion RAG technique:
    1. Generate multiple diverse queries from the original question
    2. Retrieve relevant documents for each query
    3. Apply Reciprocal Rank Fusion to combine and rank results
    4. Generate final answer using the top-ranked fused documents
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", num_samples: int = 20, num_queries: int = 3, k: int = 60, **kwargs):
        self.llm_system = FusionLLMSystem(model_name, device, num_samples=num_samples, num_queries=num_queries, technique='fusion', **kwargs)
        self.k = k
    
    def get_batch_size(self) -> int: return 20
    
    def _apply_reciprocal_rank_fusion(self, query_results: List[List[str]]) -> List[Tuple[str, float]]:
        doc_scores = defaultdict(float)
        doc_content = {}
        
        for query_idx, results in enumerate(query_results):
            for rank, doc in enumerate(results):
                doc_id = hash_string(doc)
                rrf_score = 1.0 / (self.k + rank + 1)
                doc_scores[doc_id] += rrf_score
                doc_content[doc_id] = doc
                
                logger.debug(f"Query {query_idx}, Rank {rank}: Doc {doc_id} gets RRF score {rrf_score:.4f}")
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = [doc_content[doc_id] for doc_id, _ in sorted_docs]
        
        return fused_results

    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        embedding_model = "all-MiniLM-L6-v2"
        sample = samples[0]
        if sample.get('search_results', []) != [] and \
            sample['search_results'][0].get('persistent_storage', None):
            if not hasattr(self, 'database'):
                self.database = ChunkSearcher(embedding_model=embedding_model)
                self.database.set_documents([get_storage(sample['search_results'][0]['persistent_storage'])])
            database = self.database
        else:
            documents = [[doc['page_snippet'] + "\n\n" + clean_web_content(doc.get('page_result', ''))
                        for doc in _sample.get('search_results', [])] for _sample in samples]
            database = ChunkSearcher(embedding_model=embedding_model)
            database.set_documents(documents)
        
        diverse_queries = self.llm_system.generate_diverse_queries([sample.get('question', '') for sample in samples])

        retrieved_docs = database.batch_search(
            sum(diverse_queries, []),
            sum([[i] * len(query) for i, query in enumerate(diverse_queries)], []),
            k=10
        )

        retrieved_docs = [retrieved_docs[end-len(diverse_queries[i]):end] for i, end in enumerate(np.cumsum([len(query) for query in diverse_queries]))]

        for sample, retrieved_doc in zip(samples, retrieved_docs):
            augmented_sample = sample.copy()
            if retrieved_doc: retrieved_doc = self._apply_reciprocal_rank_fusion(retrieved_doc)
            augmented_sample['context'] = ("- " + "\n- ".join(retrieved_doc))[:4000]
            result = self.llm_system.process_sample(augmented_sample)
            results.append(result)

        return results