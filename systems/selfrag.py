from systems.abstract import AbstractRAGSystem
from systems.selfllm import SelfLLMSystem
from typing import Dict, Any, List
from loguru import logger
from utils.vectordb import QdrantVectorDB  # pyright: ignore[reportMissingImports]
from utils.clean import clean_web_content  # pyright: ignore[reportMissingImports]
from utils.get_html import get_web_content  # pyright: ignore[reportMissingImports]
from utils.storage import get_storage  # pyright: ignore[reportMissingImports]


class SelfRAGSystem(AbstractRAGSystem):
    """
    Self-RAG system that implements self-reflective retrieval-augmented generation.
    
    This system implements the Self-RAG technique:
    1. Evaluate whether retrieval is needed for the query
    2. If needed, retrieve relevant documents
    3. Evaluate the relevance of retrieved documents
    4. Generate answer using retrieved context
    5. Evaluate whether the answer is supported by the context
    6. Iteratively refine if needed
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", **kwargs):
        """Initialize the Self-RAG system with an LLM and reflective retrieval."""
        # Initialize the Self-RAG LLM component
        self.llm_system = SelfLLMSystem(model_name, device, technique='self')
        self.max_iterations = kwargs.get('max_iterations', 3)
        self.relevance_threshold = kwargs.get('relevance_threshold', 0.7)
    
    def get_batch_size(self) -> int:
        """Return batch size."""
        return 1
    
    def retrieve_documents(self, query: str, documents: List[str], num_docs: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents using vector database."""
        if not documents:
            return []
            
        database = QdrantVectorDB(
            texts=documents,
            embedding_model="sentence_transformers",
            chunk_size=30,
            overlap=10
        )
        
        retrieved_docs = database.search(query, method="hybrid", k=num_docs)
        return retrieved_docs
    
    def filter_relevant_documents(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter documents based on relevance evaluation."""
        relevant_docs = []
        
        for doc in retrieved_docs:
            is_relevant = self.llm_system.evaluate_relevance(question, doc['chunk'])
            if is_relevant:
                relevant_docs.append(doc)
                logger.debug(f"Document marked as relevant: {doc['chunk'][:100]}...")
            else:
                logger.debug(f"Document marked as irrelevant: {doc['chunk'][:100]}...")
        
        logger.info(f"Filtered {len(relevant_docs)} relevant documents from {len(retrieved_docs)} retrieved")
        return relevant_docs
    
    def iterative_refinement(self, sample: Dict[str, Any], retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform iterative refinement based on self-reflection."""
        question = sample.get('question', '')
        best_result = None
        best_support_score = 'Not supported'
        
        for iteration in range(self.max_iterations):
            logger.info(f"Self-RAG iteration {iteration + 1}/{self.max_iterations}")
            
            # Create augmented sample with current context
            augmented_sample = sample.copy()
            if retrieved_docs:
                context = "- " + "\n- ".join([doc['chunk'] for doc in retrieved_docs])
                augmented_sample['search_results'] = context
                augmented_sample['technique'] = 'self'
            else:
                augmented_sample['technique'] = 'self'
            
            # Generate response
            result = self.llm_system.process_sample(augmented_sample)
            
            # Check support score
            support_score = result.get('support_score', 'Not supported')
            
            # If fully supported, we're done
            if support_score == 'Fully supported':
                logger.info(f"Achieved full support in iteration {iteration + 1}")
                best_result = result
                break
            
            # Keep track of best result so far
            if best_result is None or self._is_better_support(support_score, best_support_score):
                best_result = result
                best_support_score = support_score
            
            # If not fully supported and we have more iterations, try to refine
            if iteration < self.max_iterations - 1:
                # For simplicity, we'll use the same documents but could implement
                # query refinement or additional retrieval here
                logger.info(f"Support score '{support_score}' - continuing refinement")
        
        if best_result:
            best_result['iterations_performed'] = iteration + 1
            best_result['final_support_score'] = best_support_score
        
        return best_result or result
    
    def _is_better_support(self, score1: str, score2: str) -> bool:
        """Compare support scores to determine which is better."""
        score_rank = {
            'Fully supported': 3,
            'Partially supported': 2,
            'Not supported': 1,
            'Not evaluated': 0
        }
        return score_rank.get(score1, 0) > score_rank.get(score2, 0)
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with Self-RAG enhancement."""
        num_retrieved_docs = 10
        question = sample.get('question', '')
        
        # Step 1: Evaluate if retrieval is needed
        retrieval_needed = self.llm_system.evaluate_retrieval_need(question)
        logger.info(f"Retrieval needed: {retrieval_needed}")
        
        retrieved_docs = []
        relevant_docs = []
        
        if retrieval_needed:
            # Step 2: Prepare documents for retrieval
            documents = [
                doc['page_snippet'] + "\n\n" + clean_web_content(doc['page_result']) if doc['page_result'] else
                doc['page_snippet'] + "\n\n" + clean_web_content(get_web_content(doc['page_url']))
                for doc in sample.get('search_results', [])
            ]
            
            if documents:
                # Step 3: Retrieve documents
                retrieved_docs = self.retrieve_documents(question, documents, num_retrieved_docs)
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
                
                # Step 4: Filter relevant documents using self-reflection
                relevant_docs = self.filter_relevant_documents(question, retrieved_docs)
            else:
                logger.warning("No documents available for retrieval")
        
        # Step 5: Iterative refinement with self-reflection
        if relevant_docs or not retrieval_needed:
            result = self.iterative_refinement(sample, relevant_docs)
        else:
            # Fallback to direct processing without context
            logger.info("No relevant documents found, processing without context")
            augmented_sample = sample.copy()
            augmented_sample['technique'] = 'self'
            result = self.llm_system.process_sample(augmented_sample)
        
        # Step 6: Add Self-RAG specific information
        result.update({
            'retrieved_docs': retrieved_docs,
            'relevant_docs': relevant_docs,
            'num_retrieved_docs': len(retrieved_docs),
            'num_relevant_docs': len(relevant_docs),
            'retrieval_performed': retrieval_needed and bool(documents),
            'system_type': 'self_rag',
            'self_rag_enhanced': True,
            'retrieval_decision': {
                'needed': retrieval_needed,
                'performed': retrieval_needed and bool(documents),
                'docs_available': len(sample.get('search_results', []))
            }
        })
        
        return result
