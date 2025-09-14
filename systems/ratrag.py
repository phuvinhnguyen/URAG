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
                 max_iterations: int = 1, retrieval_k: int = 3, max_chunks_per_thought: int = 2, **kwargs):
        self.max_iterations = max_iterations  # iterations per thought
        self.retrieval_k = retrieval_k
        self.max_chunks_per_thought = max_chunks_per_thought
        self.llm_system = RATLLMSystem(
            model_name=model_name, 
            device=device, 
            technique='rat',
            temperature=kwargs.get('temperature', 0.2),  # Lower temperature for more focused revision
            **kwargs
        )
    
    def get_batch_size(self) -> int:
        return 10
    
    def _extract_queries_from_reasoning(self, reasoning_text: str, question: str) -> List[str]:
        queries = [question]  # Always include the original question
        
        # Extract reasoning steps
        lines = reasoning_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Answer|') and len(line) > 10:
                # Extract key concepts for retrieval
                # Remove numbering and common words
                cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                cleaned_line = re.sub(r'^[-*]\s*', '', cleaned_line)
                
                # Extract meaningful phrases (simple heuristic)
                if len(cleaned_line.split()) >= 3 and len(cleaned_line) > 20:
                    queries.append(cleaned_line)
        
        return queries[:5]  # Limit to top 5 queries to avoid too much retrieval
    
    def _generate_initial_cot(self, question: str) -> str:
        # Create a sample with CoT technique
        cot_sample = {'question': question, 'options': ['A', 'B', 'C', 'D']}
        
        # Temporarily set technique to CoT for initial reasoning
        original_technique = self.llm_system.technique
        self.llm_system.technique = 'cot'
        
        result = self.llm_system.process_sample(cot_sample)
        
        # Restore original technique
        self.llm_system.technique = original_technique
        
        return result['generated_response']
    
    def _revise_reasoning_with_context(self, question: str, previous_reasoning: str, 
                                     retrieved_context: str, options: List[str]) -> Dict[str, Any]:
        # Create sample with context
        revision_sample = {
            'question': question,
            'options': options,
            'context': f"Previous reasoning: {previous_reasoning}\n\nAdditional context: {retrieved_context}\n\nPlease revise your reasoning considering this new information."
        }
        
        return self.llm_system.process_sample(revision_sample)
    
    def _retrieve_chunks_for_thought(self, database: ChunkSearcher, question: str, 
                                   thought: str, doc_id: int) -> List[Dict[str, Any]]:
        """Retrieve short chunks with citations for a specific thought."""
        query = self.llm_system.generate_query_for_thought(question, thought, "")
        
        # Retrieve more documents initially
        retrieved_docs = database.search(query, doc_id, k=self.retrieval_k * 2)
        
        # Convert to chunks with citations
        chunks_with_citations = []
        for i, doc in enumerate(retrieved_docs[:self.retrieval_k]):
            # Create shorter chunks (max 200 chars for focused evidence)
            chunk_text = doc[:200] + "..." if len(doc) > 200 else doc
            chunks_with_citations.append({
                'text': chunk_text,
                'citation': f"[{i+1}]",
                'source': f"Retrieved document {i+1}"
            })
        
        return chunks_with_citations[:self.max_chunks_per_thought]
    
    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                options = sample.get('options', ['A', 'B', 'C', 'D'])
                query_time = sample.get('query_time', 'March 1, 2025')
                
                # Step 1: Generate initial chain-of-thoughts
                current_reasoning = self._generate_initial_cot(question)
                reasoning_history = [{"iteration": 0, "reasoning": current_reasoning, "type": "initial"}]
                citations = []
                
                # Step 2: Extract thoughts and revise each one individually
                thoughts = self.llm_system.extract_thoughts(current_reasoning)
                
                thought_revisions = []
                for thought_idx, thought in enumerate(thoughts):
                    # Retrieve chunks for this specific thought
                    chunks_with_citations = self._retrieve_chunks_for_thought(
                        database, question, thought, doc_id
                    )
                    
                    if chunks_with_citations:
                        # Format evidence with citations
                        evidence_text = "\n".join([
                            f"{chunk['citation']} {chunk['text']}"
                            for chunk in chunks_with_citations
                        ])
                        
                        # Collect citations for final answer
                        citations.extend([chunk['citation'] + ": " + chunk['source'] 
                                        for chunk in chunks_with_citations])
                        
                        # Revise this specific thought
                        for iteration in range(self.max_iterations):
                            revised_reasoning = self.llm_system.revise_thought_with_evidence(
                                question, current_reasoning, thought, evidence_text, options
                            )
                            
                            # Update current reasoning with the revision
                            current_reasoning = revised_reasoning
                            
                            thought_revisions.append({
                                "thought_idx": thought_idx,
                                "iteration": iteration + 1,
                                "original_thought": thought,
                                "evidence": evidence_text,
                                "revised_reasoning": revised_reasoning,
                                "chunks": chunks_with_citations
                            })
                    else:
                        # No evidence found for this thought
                        thought_revisions.append({
                            "thought_idx": thought_idx,
                            "iteration": 0,
                            "original_thought": thought,
                            "evidence": "No relevant evidence found",
                            "revised_reasoning": current_reasoning,
                            "chunks": []
                        })
                
                # Step 3: Generate final answer
                final_sample = {
                    'question': question,
                    'options': options,
                    'context': f"Revised reasoning: {current_reasoning}\nQuery Time: {query_time}"
                }
                
                final_result = self.llm_system.process_sample(final_sample)
                
                # Add citations to the response
                final_response = final_result['generated_response']
                if citations:
                    unique_citations = list(dict.fromkeys(citations))  # Remove duplicates
                    citation_text = "\n\nSources:\n" + "\n".join(unique_citations[:5])  # Limit citations
                    final_response += citation_text
                
                # Combine results
                result = {
                    'id': sample.get('id', 'unknown'),
                    'generated_response': final_response,
                    'predicted_answer': final_result['predicted_answer'],
                    'conformal_probabilities': final_result['conformal_probabilities'],
                    'reasoning_history': reasoning_history,
                    'thought_revisions': thought_revisions,
                    'technique': 'rat_rag',
                    'total_thoughts': len(thoughts),
                    'total_chunks_used': sum(len(rev.get('chunks', [])) for rev in thought_revisions),
                    'citations': unique_citations[:5] if citations else []
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
