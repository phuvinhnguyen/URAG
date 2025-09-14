from systems.abstract import AbstractRAGSystem
from systems.ratllm import RATLLMSystem
from typing import Dict, Any, List
from utils.clean import clean_web_content
from utils.ramdb import ChunkSearcher
from utils.storage import get_storage
import re

class RATRAGSystem(AbstractRAGSystem):
    """
    RAT RAG system that combines retrieval-augmented generation with iterative thought refinement.
    
    This system implements the full RAT technique with retrieval:
    1. Generate initial chain-of-thoughts
    2. For each reasoning step, retrieve relevant documents
    3. Iteratively revise reasoning with retrieved information
    4. Generate final answer with enhanced context
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda", 
                 max_iterations: int = 2, **kwargs):
        """Initialize RAT RAG system with LLM and retrieval components."""
        self.max_iterations = max_iterations
        self.llm_system = RATLLMSystem(
            model_name=model_name, 
            device=device, 
            max_iterations=max_iterations,
            **kwargs
        )
    
    def get_batch_size(self) -> int:
        return 10
    
    def _extract_queries_from_reasoning(self, reasoning_text: str, question: str) -> List[str]:
        """Extract search queries from reasoning steps and original question."""
        queries = [question]  # Always include the original question
        
        # Extract reasoning steps
        lines = reasoning_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Answer|'):
                # Extract key concepts for retrieval
                # Remove numbering and common words
                cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                cleaned_line = re.sub(r'^[-*]\s*', '', cleaned_line)
                
                # Extract meaningful phrases (simple heuristic)
                if len(cleaned_line.split()) >= 3 and len(cleaned_line) > 20:
                    queries.append(cleaned_line)
        
        return queries[:5]  # Limit to top 5 queries to avoid too much retrieval
    
    def _retrieve_for_reasoning_step(self, database: ChunkSearcher, question: str, 
                                   reasoning_text: str, doc_id: int) -> List[str]:
        """Retrieve relevant documents for a reasoning step."""
        queries = self._extract_queries_from_reasoning(reasoning_text, question)
        
        all_retrieved = []
        for query in queries:
            retrieved = database.search(query, doc_id, k=3)  # Fewer docs per query
            all_retrieved.extend(retrieved)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_retrieved = []
        for doc in all_retrieved:
            if doc not in seen:
                seen.add(doc)
                unique_retrieved.append(doc)
        
        return unique_retrieved[:8]  # Limit total retrieved documents
    
    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of samples using RAT with retrieval."""
        results = []
        embedding_model = "all-MiniLM-L6-v2"
        
        # Setup database similar to SimpleRAG
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
        
        for doc_id, sample in enumerate(samples):
            try:
                question = sample.get('question', '')
                options = sample.get('options', ['A', 'B', '', 'D'])
                query_time = sample.get('query_time', 'March 1, 2025')
                
                # Step 1: Generate initial chain-of-thoughts (similar to RATLLMSystem)
                initial_reasoning = self.llm_system._generate_initial_cot(question)
                current_reasoning = initial_reasoning
                
                reasoning_history = [{"iteration": 0, "reasoning": initial_reasoning, "type": "initial"}]
                
                # Step 2: Iteratively refine with retrieval
                for iteration in range(1, self.max_iterations + 1):
                    # Retrieve documents based on current reasoning
                    retrieved_docs = self._retrieve_for_reasoning_step(
                        database, question, current_reasoning, doc_id
                    )
                    
                    # Format retrieved information
                    if retrieved_docs:
                        retrieved_info = ("Retrieved context:\n- " + 
                                        "\n- ".join(retrieved_docs)[:3000] + 
                                        f'\nQuery Time: {query_time}')
                    else:
                        retrieved_info = f"No additional context found. Query Time: {query_time}"
                    
                    # Revise reasoning with retrieved information
                    revised_response, conformal_probabilities = self.llm_system._revise_reasoning(
                        question, current_reasoning, retrieved_info, options
                    )
                    
                    current_reasoning = revised_response
                    reasoning_history.append({
                        "iteration": iteration,
                        "reasoning": revised_response,
                        "retrieved_info": retrieved_info,
                        "retrieved_docs": retrieved_docs,
                        "type": "revision_with_retrieval"
                    })
                
                # Step 3: Generate final answer with all context
                final_context = "\n\n".join([
                    step.get("retrieved_info", "") for step in reasoning_history[1:]
                    if step.get("retrieved_info")
                ])
                
                final_prompt = (f"{question}\n\n"
                              f"Final reasoning with context: {current_reasoning}\n\n"
                              f"Additional context: {final_context[:2000]}\n\n"
                              f"Please provide your final answer in the format Answer|X where X is your answer (A, B, C, D, ...)")
                
                final_response, final_probabilities = self.llm_system._generate_response_with_probabilities(
                    final_prompt, options
                )
                
                result = {
                    'id': sample.get('id', 'unknown'),
                    'generated_response': final_response,
                    'predicted_answer': max(final_probabilities.items(), key=lambda x: x[1])[0],
                    'conformal_probabilities': final_probabilities,
                    'reasoning_history': reasoning_history,
                    'technique': 'rat_rag',
                    'iterations_used': self.max_iterations,
                    'total_retrieved_docs': sum(len(step.get("retrieved_docs", [])) for step in reasoning_history)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                # Fallback to simple LLM processing
                try:
                    fallback_result = self.llm_system.process_sample(sample)
                    fallback_result['technique'] = 'rat_rag_fallback'
                    results.append(fallback_result)
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
        
        return results
