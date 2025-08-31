from systems.abstract import AbstractRAGSystem
from systems.hydellm import HyDELLMSystem
from typing import Dict, Any, List
from loguru import logger
import re

from utils.vectordb import QdrantVectorDB  # pyright: ignore[reportMissingImports]
from utils.clean import clean_web_content  # pyright: ignore[reportMissingImports]
from utils.get_html import get_web_content  # pyright: ignore[reportMissingImports]



class HyDERAGSystem(AbstractRAGSystem):
    """
    HyDE RAG system that uses hypothetical document embeddings for retrieval.
    
    This system implements the HyDE (Hypothetical Document Embeddings) technique:
    1. Generate a hypothetical document that would answer the query
    2. Use that hypothetical document for retrieval instead of the original query
    3. Retrieve relevant documents based on hypothetical document similarity
    4. Generate final answer using retrieved context
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", **kwargs):
        """Initialize the HyDE RAG system with an LLM and enhanced retrieval."""
        # Initialize the HyDE LLM component
        self.llm_system = HyDELLMSystem(model_name, device, technique='rag')
    
    def get_batch_size(self) -> int:
        """Return batch size."""
        return 1
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with HyDE RAG enhancement."""
        num_retrieved_docs = 10
        question = sample.get('question', '')
        documents = [doc['page_snippet'] + "\n\n" + clean_web_content(doc['page_result']) if doc['page_result'] else
                     doc['page_snippet'] + "\n\n" + clean_web_content(get_web_content(doc['page_url']))
                     for doc in sample.get('search_results', [])]
        
        database = QdrantVectorDB(
            texts=documents,
            embedding_model="sentence_transformers",
            chunk_size=30,
            overlap=10
        )
        
        # Step 1: Generate hypothetical document using HyDE LLM
        hypothetical_doc = self.llm_system.generate_hypothetical_document(question)
        logger.info(f"Generated hypothetical document: {hypothetical_doc[:100]}...")

        hyde_retrieved_docs = database.search(hypothetical_doc, method="hybrid", k=num_retrieved_docs)
        
        # Step 4: Augment sample with retrieved context
        augmented_sample = sample.copy()
        if hyde_retrieved_docs:
            augmented_sample['search_results'] = "- " + "\n- ".join([i['chunk'] for i in hyde_retrieved_docs])
            augmented_sample['technique'] = 'hyde'            
        else:
            augmented_sample['technique'] = 'hyde'
        
        # Pass the hypothetical document to avoid regenerating it in HyDELLMSystem
        augmented_sample['hypothetical_document'] = hypothetical_doc
        
        # Step 5: Process through HyDE LLM system
        result = self.llm_system.process_sample(augmented_sample)
        
        # Step 6: Add HyDE-specific information
        result.update({
            'retrieved_docs': hyde_retrieved_docs,
            'num_retrieved_docs': len(hyde_retrieved_docs),
            'hyde_enhanced': bool(hyde_retrieved_docs),
            'system_type': 'hyde_rag',
            'hypothetical_document': hypothetical_doc,
        })
        
        return result
