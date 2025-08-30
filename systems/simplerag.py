from systems.abstract import AbstractRAGSystem
from systems.simplellm import SimpleLLMSystem
from typing import Dict, Any, List
from loguru import logger
from utils.clean import clean_web_content
from utils.vectordb import QdrantVectorDB
from utils.get_html import get_web_content


class SimpleRAGSystem(AbstractRAGSystem):
    """
    Simple RAG system that performs keyword-based retrieval and augmentation.
    
    This demonstrates how to implement a traditional RAG system with retrieval.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", device: str = "auto", **kwargs):
        """Initialize the RAG system with an LLM and simple retrieval."""
        # Initialize the LLM component
        self.llm_system = SimpleLLMSystem(model_name, device)
    
    def get_batch_size(self) -> int:
        """Return batch size."""
        return 1
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with RAG enhancement."""
        # Retrieve relevant context
        question = sample.get('question', '')
        documents = [doc['page_snippet'] + "\n\n" + clean_web_content(doc['page_result']) if doc['page_result'] else
                     doc['page_snippet'] + "\n\n" + clean_web_content(get_web_content(doc['page_url']))
                     for doc in sample.get('search_results', [])]
        
        database = QdrantVectorDB(
            texts=documents,
            embedding_model="sentence_transformers",
            chunk_size=70,
            overlap=20
        )
                
        retrieved_docs = database.search(question, method="hybrid", k=3)

        # clean and remove database from memory to save RAM as much as possible
        try:
            database.client.delete_collection(database.collections[0])
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
        del database
        
        # Augment sample with retrieved context
        augmented_sample = sample.copy()
        if retrieved_docs:
            # Combine retrieved documents
            retrieved_context = "\n".join([i['chunk'] for i in retrieved_docs])
            
            # Add to existing context if any
            existing_context = sample.get('search_results', sample.get('context', ''))
            if existing_context:
                combined_context = f"{existing_context}\n\nAdditional context:\n{retrieved_context}"
            else:
                combined_context = f"Context:\n{retrieved_context}"
            
            augmented_sample['search_results'] = combined_context
            augmented_sample['technique'] = 'rag'
            
            logger.debug(f"Enhanced sample with {len(retrieved_docs)} retrieved documents")
        else:
            logger.debug("No relevant documents found for retrieval")
        
        # Process through LLM
        result = self.llm_system.process_sample(augmented_sample)
        
        # Add RAG-specific information
        result.update({
            'retrieved_docs': retrieved_docs,
            'num_retrieved_docs': len(retrieved_docs),
            'rag_enhanced': bool(retrieved_docs),
            'system_type': 'simple_rag'
        })
        
        return result
