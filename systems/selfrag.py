from systems.abstract import AbstractRAGSystem
from systems.selfllm import SelfLLMSystem
from typing import Dict, Any, List
from utils.clean import clean_web_content
from utils.ramdb import ChunkSearcher
from utils.storage import get_storage
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

class SelfRAGSystem(AbstractRAGSystem):
    """
    Self-RAG system that combines adaptive retrieval with self-reflection capabilities.
    
    This system implements the Self-RAG framework:
    1. Uses reflection tokens to decide when to retrieve
    2. Retrieves relevant passages when needed
    3. Generates responses with self-critique using reflection tokens
    4. Supports segment-wise beam search for optimal responses
    
    Based on the original Self-RAG paper: https://arxiv.org/abs/2310.11511
    """
    
    def __init__(self, model_name: str = "selfrag/selfrag_llama2_7b", device: str = "cuda", 
                 threshold: float = 0.2, max_depth: int = 6, beam_width: int = 2,
                 w_rel: float = 1.0, w_sup: float = 1.0, w_use: float = 0.5, **kwargs):
        
        # Initialize the underlying LLM system
        self.llm_system = SelfLLMSystem(
            model_name=model_name, 
            device=device, 
            technique='selfrag', 
            threshold=threshold, 
            **kwargs
        )
        
        self.threshold = threshold
        self.max_depth = max_depth
        self.beam_width = beam_width
        
        # Weights for scoring different reflection aspects
        self.w_rel = w_rel  # Weight for relevance score
        self.w_sup = w_sup  # Weight for support score  
        self.w_use = w_use  # Weight for utility score
        
        # For retrieval
        self.database = None
        self.embedding_model = "all-MiniLM-L6-v2"
        
        print(f"[SELFRAG] System initialized with model: {model_name}")
    
    def get_batch_size(self) -> int: return 40
    
    def _setup_database(self, samples: List[Dict[str, Any]]):
        """Setup retrieval database from samples following the pattern from other RAG systems."""
        if not samples:
            return
            
        sample = samples[0]
        
        # Check if we have persistent storage (like in HyDERAG)
        if (sample.get('search_results', []) and 
            sample['search_results'][0].get('persistent_storage', None)):
            
            if not hasattr(self, 'database') or self.database is None:
                self.database = ChunkSearcher(embedding_model=self.embedding_model)
                self.database.set_documents([
                    get_storage(sample['search_results'][0]['persistent_storage'])
                ])
        else:
            # Build database from search results (like in HyDERAG)
            documents = []
            for _sample in samples:
                sample_docs = []
                for doc in _sample.get('search_results', []):
                    content = doc.get('page_snippet', '') + "\n\n" + clean_web_content(doc.get('page_result', ''))
                    sample_docs.append(content)
                documents.append(sample_docs)
            
            self.database = ChunkSearcher(embedding_model=self.embedding_model)
            self.database.set_documents(documents)
    
    def _adaptive_retrieve(self, question: str, sample_id: int, options: List[str]) -> List[str]:
        """Perform adaptive retrieval based on Self-RAG decision."""
        # Check if model decides to retrieve
        should_retrieve = self.llm_system.make_retrieval_decision(question, options)
        
        print(f"[SELFRAG] Retrieval decision for sample {sample_id}: {'RETRIEVE' if should_retrieve else 'NO RETRIEVE'}")
        
        if not should_retrieve:
            return []
        
        # Retrieve relevant documents
        if self.database is None:
            return []
        
        retrieved_docs = self.database.search(question, sample_id, k=5)
        print(f"[SELFRAG] Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs
    
    def _segment_wise_beam_search(self, sample: Dict[str, Any], retrieved_docs: List[str]) -> Dict[str, Any]:
        """
        Implement segment-wise beam search for Self-RAG.
        
        This generates multiple candidate responses and selects the best one
        based on reflection token scores.
        """
        candidates = []
        
        # Generate candidates with different contexts (following the Self-RAG approach)
        contexts_to_try = [
            "",  # No context
            "\n".join(retrieved_docs[:3]) if retrieved_docs else "",  # Top 3 docs
            "\n".join(retrieved_docs) if retrieved_docs else "",  # All docs
        ]
        
        for i, context in enumerate(contexts_to_try):
            if context or not retrieved_docs:  # Always try no context, or if no docs retrieved
                augmented_sample = sample.copy()
                augmented_sample['context'] = context
                
                # Generate response
                result = self.llm_system.process_sample(augmented_sample)
                
                # Extract reflection token scores from the result
                reflection_tokens = result.get('reflection_tokens', {})
                
                # Calculate composite score using reflection tokens (Self-RAG scoring)
                relevance_score = 0.5  # Default
                if reflection_tokens.get('relevance') == 'relevant':
                    relevance_score = 1.0
                elif reflection_tokens.get('relevance') == 'irrelevant':
                    relevance_score = 0.0
                    
                support_score = 0.5  # Default
                if reflection_tokens.get('support') == 'supported':
                    support_score = 1.0
                elif reflection_tokens.get('support') == 'partially_supported':
                    support_score = 0.7
                elif reflection_tokens.get('support') == 'no_support':
                    support_score = 0.0
                    
                utility_score = reflection_tokens.get('utility', 3) / 5.0  # Convert to 0-1 scale
                
                # Self-RAG composite scoring
                composite_score = (self.w_rel * relevance_score + 
                                 self.w_sup * support_score + 
                                 self.w_use * utility_score)
                
                candidates.append({
                    'result': result,
                    'score': composite_score,
                    'context_length': len(context),
                    'relevance_score': relevance_score,
                    'support_score': support_score,
                    'utility_score': utility_score
                })
        
        # Select best candidate based on Self-RAG scoring
        if candidates:
            best_candidate = max(candidates, key=lambda x: x['score'])
            best_result = best_candidate['result']
            
            # Add Self-RAG specific metadata
            best_result['composite_score'] = best_candidate['score']
            best_result['context_length'] = best_candidate['context_length']
            best_result['relevance_score'] = best_candidate['relevance_score']
            best_result['support_score'] = best_candidate['support_score']
            best_result['utility_score'] = best_candidate['utility_score']
            
            return best_result
        else:
            # Fallback to basic processing
            return self.llm_system.process_sample(sample)
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample with Self-RAG adaptive retrieval."""
        question = sample.get('question', '')
        options = sample.get('options', [])
        
        # Setup database if needed (for single sample processing)
        if self.database is None and sample.get('search_results'):
            self._setup_database([sample])
        
        # Adaptive retrieval using Self-RAG reflection tokens
        retrieved_docs = self._adaptive_retrieve(question, 0, options)
        
        # Use segment-wise beam search to find best response
        result = self._segment_wise_beam_search(sample, retrieved_docs)
        
        # Add retrieval information
        result['retrieved_docs'] = retrieved_docs
        result['num_retrieved'] = len(retrieved_docs)
        
        return result
    
    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of samples with Self-RAG efficiently using true batch processing."""
        results = []
        
        # Setup database for batch processing
        self._setup_database(samples)
        
        print(f"[SELFRAG] Processing batch of {len(samples)} samples")
        
        # Step 1: Batch make retrieval decisions for all samples
        retrieval_decisions = []
        questions = [sample.get('question', '') for sample in samples]
        options_list = [sample.get('options', []) for sample in samples]
        
        retrieval_decisions = self._batch_make_retrieval_decisions(questions, options_list)
        
        # Step 2: Batch retrieve for samples that need retrieval
        all_retrieved_docs = self._batch_retrieve_documents(questions, retrieval_decisions)
        
        # Step 3: Batch process all samples with beam search
        try:
            results = self._batch_segment_wise_beam_search(samples, all_retrieved_docs, retrieval_decisions)
        except Exception as e:
            logger.exception(f"Batch processing failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            results = self._sequential_fallback_processing(samples, all_retrieved_docs, retrieval_decisions)
        
        return results
    
    def _batch_make_retrieval_decisions(self, questions: List[str], options_list: List[List[str]]) -> List[bool]:
        """Batch make retrieval decisions for multiple questions."""
        # Create batch prompts for retrieval decision
        retrieval_prompts = []
        for question, options in zip(questions, options_list):
            system_message = """You are a helpful assistant that decides whether to retrieve information. Use these reflection tokens:
- <|retrieve|> when you need external information to answer the question
- <|no_retrieve|> when you can answer the question directly with your knowledge

Respond with only the appropriate reflection token."""
            user_message = f"Question: {question}\nOptions: {', '.join(options)}\n\nDo you need to retrieve information to answer this question?"
            prompt = self.llm_system._create_unified_prompt(system_message, user_message)
            retrieval_prompts.append(prompt)
        
        # Batch generate retrieval decisions
        try:
            # Use batch processing if available
            batch_responses, _ = self.llm_system._generate_response_with_probabilities(retrieval_prompts, [['retrieve', 'no_retrieve']] * len(retrieval_prompts))
            
            retrieval_decisions = []
            for response in batch_responses:
                reflection_tokens = self.llm_system.extract_reflection_tokens(response)
                should_retrieve = reflection_tokens.get('retrieval') == 'retrieve'
                retrieval_decisions.append(should_retrieve)
            
            return retrieval_decisions
            
        except Exception as e:
            logger.warning(f"Batch retrieval decision failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            retrieval_decisions = []
            for question, options in zip(questions, options_list):
                should_retrieve = self.llm_system.make_retrieval_decision(question, options)
                retrieval_decisions.append(should_retrieve)
            return retrieval_decisions
    
    def _batch_retrieve_documents(self, questions: List[str], retrieval_decisions: List[bool]) -> List[List[str]]:
        """Batch retrieve documents for questions that need retrieval."""
        all_retrieved_docs = []
        
        if self.database is not None:
            # Collect queries for samples that need retrieval
            queries_to_retrieve = []
            sample_indices = []
            
            for i, (should_retrieve, question) in enumerate(zip(retrieval_decisions, questions)):
                if should_retrieve:
                    queries_to_retrieve.append(question)
                    sample_indices.append(i)
            
            if queries_to_retrieve:
                # Batch retrieval
                batch_retrieved = self.database.batch_search(
                    queries_to_retrieve, 
                    sample_indices, 
                    k=5
                )
                
                # Map back to all samples
                retrieval_map = dict(zip(sample_indices, batch_retrieved))
                for i in range(len(questions)):
                    if i in retrieval_map:
                        all_retrieved_docs.append(retrieval_map[i])
                    else:
                        all_retrieved_docs.append([])
            else:
                all_retrieved_docs = [[] for _ in questions]
        else:
            all_retrieved_docs = [[] for _ in questions]
        
        return all_retrieved_docs
    
    def _batch_segment_wise_beam_search(self, samples: List[Dict[str, Any]], all_retrieved_docs: List[List[str]], retrieval_decisions: List[bool]) -> List[Dict[str, Any]]:
        """Batch process samples with segment-wise beam search."""
        results = []
        
        # Group samples by context strategy for batch processing
        batch_contexts = []
        sample_indices = []
        
        for i, (sample, retrieved_docs) in enumerate(zip(samples, all_retrieved_docs)):
            # Generate different context strategies for this sample
            contexts_to_try = [
                "",  # No context
                "\n".join(retrieved_docs[:3]) if retrieved_docs else "",  # Top 3 docs
                "\n".join(retrieved_docs) if retrieved_docs else "",  # All docs
            ]
            
            for context in contexts_to_try:
                if context or not retrieved_docs:  # Always try no context
                    augmented_sample = sample.copy()
                    augmented_sample['context'] = context
                    batch_contexts.append(augmented_sample)
                    sample_indices.append(i)
        
        # Batch process all contexts
        if batch_contexts:
            try:
                batch_results = self.llm_system.batch_process_samples(batch_contexts)
                
                # Group results back by original sample and select best
                sample_results = {}
                for idx, result in zip(sample_indices, batch_results):
                    if idx not in sample_results:
                        sample_results[idx] = []
                    sample_results[idx].append(result)
                
                # Select best result for each sample using Self-RAG scoring
                for i in range(len(samples)):
                    if i in sample_results:
                        candidates = sample_results[i]
                        best_result = self._select_best_candidate_from_batch(candidates, all_retrieved_docs[i])
                        
                        # Add retrieval information
                        best_result['retrieved_docs'] = all_retrieved_docs[i]
                        best_result['num_retrieved'] = len(all_retrieved_docs[i])
                        best_result['retrieval_decision'] = retrieval_decisions[i]
                        
                        results.append(best_result)
                    else:
                        # Fallback for samples with no results
                        fallback_result = self.llm_system.process_sample(samples[i])
                        fallback_result['retrieved_docs'] = all_retrieved_docs[i]
                        fallback_result['num_retrieved'] = len(all_retrieved_docs[i])
                        fallback_result['retrieval_decision'] = retrieval_decisions[i]
                        results.append(fallback_result)
                        
            except Exception as e:
                logger.exception(f"Batch beam search failed: {e}")
                raise e
        else:
            # No contexts to process, use basic processing
            results = self._sequential_fallback_processing(samples, all_retrieved_docs, retrieval_decisions)
        
        return results
    
    def _select_best_candidate_from_batch(self, candidates: List[Dict[str, Any]], retrieved_docs: List[str]) -> Dict[str, Any]:
        """Select the best candidate from batch results using Self-RAG scoring."""
        if not candidates:
            return {}
        
        if len(candidates) == 1:
            return candidates[0]
        
        scored_candidates = []
        
        for candidate in candidates:
            # Extract reflection token scores from the result (if available)
            reflection_tokens = candidate.get('reflection_tokens', {})
            
            # Calculate composite score using reflection tokens (Self-RAG scoring)
            relevance_score = 0.5  # Default
            if reflection_tokens.get('relevance') == 'relevant':
                relevance_score = 1.0
            elif reflection_tokens.get('relevance') == 'irrelevant':
                relevance_score = 0.0
                
            support_score = 0.5  # Default
            if reflection_tokens.get('support') == 'supported':
                support_score = 1.0
            elif reflection_tokens.get('support') == 'partially_supported':
                support_score = 0.7
            elif reflection_tokens.get('support') == 'no_support':
                support_score = 0.0
                
            utility_score = reflection_tokens.get('utility', 3) / 5.0  # Convert to 0-1 scale
            
            # Self-RAG composite scoring
            composite_score = (self.w_rel * relevance_score + 
                             self.w_sup * support_score + 
                             self.w_use * utility_score)
            
            scored_candidates.append({
                'result': candidate,
                'score': composite_score,
                'relevance_score': relevance_score,
                'support_score': support_score,
                'utility_score': utility_score
            })
        
        # Select best candidate based on Self-RAG scoring
        best_candidate = max(scored_candidates, key=lambda x: x['score'])
        best_result = best_candidate['result']
        
        # Add Self-RAG specific metadata
        best_result['composite_score'] = best_candidate['score']
        best_result['relevance_score'] = best_candidate['relevance_score']
        best_result['support_score'] = best_candidate['support_score']
        best_result['utility_score'] = best_candidate['utility_score']
        
        return best_result
    
    def _sequential_fallback_processing(self, samples: List[Dict[str, Any]], all_retrieved_docs: List[List[str]], retrieval_decisions: List[bool]) -> List[Dict[str, Any]]:
        """Fallback to sequential processing when batch processing fails."""
        results = []
        
        for i, (sample, retrieved_docs) in enumerate(zip(samples, all_retrieved_docs)):
            try:
                # Use segment-wise beam search
                result = self._segment_wise_beam_search(sample, retrieved_docs)
                
                # Add retrieval information
                result['retrieved_docs'] = retrieved_docs
                result['num_retrieved'] = len(retrieved_docs)
                result['retrieval_decision'] = retrieval_decisions[i]
                
                results.append(result)
                
            except Exception as e:
                logger.exception(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                # Fallback to basic LLM processing
                try:
                    fallback_result = self.llm_system.process_sample(sample)
                    fallback_result['retrieved_docs'] = []
                    fallback_result['num_retrieved'] = 0
                    fallback_result['retrieval_decision'] = False
                    fallback_result['error'] = str(e)
                    results.append(fallback_result)
                except Exception as e2:
                    logger.exception(f"Fallback also failed for sample {sample.get('id', 'unknown')}: {e2}")
                    # Last resort: return minimal result
                    results.append({
                        'id': sample.get('id', 'unknown'),
                        'generated_response': 'Error processing sample',
                        'predicted_answer': 'A',
                        'conformal_probabilities': {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25},
                        'technique': 'selfrag',
                        'retrieval_decision': False,
                        'error': str(e2)
                    })
        
        return results
