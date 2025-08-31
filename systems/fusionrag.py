from systems.abstract import AbstractRAGSystem
from systems.fusionllm import FusionLLMSystem
from typing import Dict, Any, List, Tuple
from loguru import logger
import re
from collections import defaultdict, Counter
import math
from utils.clean import clean_web_content  # pyright: ignore[reportMissingImports]
from utils.get_html import get_web_content  # pyright: ignore[reportMissingImports]
from utils.vectordb import QdrantVectorDB  # pyright: ignore[reportMissingImports]
import hashlib

def hash_string(s):
    """Hash a string using SHA-256"""
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
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", num_queries: int = 3, k: int = 60, **kwargs):
        """Initialize the Fusion RAG system with an LLM and enhanced retrieval."""
        # Initialize the Fusion LLM component
        self.llm_system = FusionLLMSystem(model_name, device, num_queries=num_queries, technique='fusion')
        self.k = k  # RRF parameter (higher values reduce the impact of high-ranked documents)
        
    def get_batch_size(self) -> int:
        """Return batch size."""
        return 1
    
    def _apply_reciprocal_rank_fusion(self, query_results: List[List[Dict[str, Any]]]) -> List[Tuple[str, float]]:
        """Apply Reciprocal Rank Fusion to combine results from multiple queries."""
        # Document ID to (document, total_rrf_score)
        doc_scores = defaultdict(float)
        doc_content = {}
        
        for query_idx, results in enumerate(query_results):
            for rank, item in enumerate(results):
                doc = item['chunk']
                doc_id = hash_string(doc)
                # RRF formula: 1 / (k + rank) where k is typically 60
                rrf_score = 1.0 / (self.k + rank + 1)  # +1 because rank is 0-indexed
                doc_scores[doc_id] += rrf_score
                doc_content[doc_id] = item
                
                logger.debug(f"Query {query_idx}, Rank {rank}: Doc {doc_id} gets RRF score {rrf_score:.4f}")
        
        # Sort by total RRF score (descending)
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return (document_content, final_rrf_score)
        fused_results = [doc_content[doc_id] for doc_id, score in sorted_docs]
        
        return fused_results
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with Fusion RAG enhancement."""
        num_retrieved_docs = 10

        question = sample.get('question', '')
        documents = [doc['page_snippet'] + "\n\n" + clean_web_content(doc['page_result']) if doc['page_result'] else
                     doc['page_snippet'] + "\n\n" + clean_web_content(get_web_content(doc['page_url']))
                     for doc in sample.get('search_results', [])]
        
        # Step 1: Generate diverse queries using Fusion LLM
        diverse_queries = self.llm_system.generate_diverse_queries(question)

        database = QdrantVectorDB(
            texts=documents,
            embedding_model="sentence_transformers",
            chunk_size=30,
            overlap=10
        )
        
        retrieved_docs = [sorted(database.search(query, method="hybrid", k=num_retrieved_docs), key=lambda x: x['score'], reverse=True) for query in diverse_queries]
        
        retrieved_docs = self._apply_reciprocal_rank_fusion(retrieved_docs)[:num_retrieved_docs]

        # Step 4: Augment sample with retrieved context
        augmented_sample = sample.copy()
        augmented_sample['diverse_queries'] = diverse_queries 
        if retrieved_docs:
            augmented_sample['search_results'] = "- " + "\n- ".join([i['chunk'] for i in retrieved_docs])
            augmented_sample['technique'] = 'fusion'
        
        # Step 5: Process through Fusion LLM system
        result = self.llm_system.process_sample(augmented_sample)
        
        # Step 6: Add Fusion-specific information
        result.update({
            'retrieved_docs': retrieved_docs,
            'num_retrieved_docs': len(retrieved_docs),
            'fusion_enhanced': bool(retrieved_docs),
            'system_type': 'fusion_rag',
            'diverse_queries': diverse_queries,
            'rrf_k_parameter': self.k
        })
        
        return result
