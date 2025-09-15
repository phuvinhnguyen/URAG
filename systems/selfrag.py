from systems.abstract import AbstractRAGSystem
from systems.selfllm import SelfLLMSystem
from typing import Dict, Any, List
from loguru import logger
from utils.ramdb import ChunkSearcher  # pyright: ignore[reportMissingImports]
from utils.clean import clean_web_content  # pyright: ignore[reportMissingImports]
from utils.storage import get_storage  # pyright: ignore[reportMissingImports]

class SelfRAGSystem(AbstractRAGSystem):
    """
    Enhanced Self-RAG system that implements self-reflective retrieval-augmented generation.
    
    This system implements the complete Self-RAG technique with improvements:
    1. Evaluate whether retrieval is needed for the query using <retrieve> tokens
    2. If needed, retrieve relevant documents and filter using <relevant> tokens
    3. Generate answers segment-by-segment with inline reflection tokens
    4. Evaluate support using <support> tokens (Fully/Partially/Not supported)
    5. Evaluate utility using <utility> tokens (1-5 scale)
    6. Iteratively refine with additional retrieval if utility/support is low
    7. Maintain batch processing efficiency throughout all operations
    
    Key improvements over basic RAG:
    - Adaptive retrieval based on question complexity
    - Segment-level generation with reflection
    - Iterative refinement with quality thresholds
    - Comprehensive evaluation (relevance, support, utility)
    - Optimized batch processing for performance
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", **kwargs):
        """Initialize the Self-RAG system with an LLM and reflective retrieval."""
        self.max_iterations = kwargs.pop('max_iterations', 3)
        self.relevance_threshold = kwargs.pop('relevance_threshold', 0.7)
        self.utility_threshold = kwargs.pop('utility_threshold', 3)  # Minimum utility score
        self.max_segments = kwargs.pop('max_segments', 3)  # Maximum segments per generation
        self.llm_system = SelfLLMSystem(model_name, device, technique='rag', **kwargs)
    
    def get_batch_size(self) -> int: return 20
    
    def filter_relevant_documents(self, questions: List[str], retrieved_docs: List[List[str]]) -> List[Dict[str, Any]]:
        """Filter documents based on relevance evaluation."""
        relevant_docs = [[] for _ in questions]
        is_relevant_batch = self.llm_system.evaluate_relevance(questions, retrieved_docs)
        
        for i, (is_relevant, docs) in enumerate(zip(is_relevant_batch, retrieved_docs)):
            relevant_count = 0
            for is_relevant_doc, doc in zip(is_relevant, docs):
                if is_relevant_doc:
                    relevant_docs[i].append(doc)
                    relevant_count += 1
            
            # Log document filtering results
            logger.info(f"[SELFRAG] Document filtering for question {i+1}:")
            logger.info(f"[SELFRAG] Total docs retrieved: {len(docs)}")
            logger.info(f"[SELFRAG] Relevant docs found: {relevant_count}")
            logger.info(f"[SELFRAG] Filtered docs: {len(docs) - relevant_count}")
            logger.info("---")
        
        return relevant_docs
    
    def iterative_retrieve_and_refine(self, questions: List[str], database, initial_contexts: List[str] = None) -> List[Dict[str, Any]]:
        """Perform iterative retrieval and refinement for better answers with optimized batching."""
        if initial_contexts is None:
            initial_contexts = [""] * len(questions)
        
        # Track active questions and their contexts
        active_questions = list(questions)
        active_contexts = list(initial_contexts)
        active_indices = list(range(len(questions)))
        final_results = [None] * len(questions)
        
        for iteration in range(self.max_iterations):
            if not active_questions:
                break
            
            logger.info(f"[SELFRAG] === Iteration {iteration + 1}/{self.max_iterations} ===")
            logger.info(f"[SELFRAG] Active questions: {len(active_questions)}")
            
            # Generate segments for all active questions in batch
            segment_results = self.llm_system.generate_with_segments(active_questions, active_contexts, self.max_segments)
            
            # Identify questions that need more retrieval
            questions_needing_retrieval = []
            contexts_needing_retrieval = []
            indices_needing_retrieval = []
            
            # Process results and identify completed questions
            new_active_questions = []
            new_active_contexts = []
            new_active_indices = []
            
            for i, (segment_result, orig_idx) in enumerate(zip(segment_results, active_indices)):
                # Check if this question is done (good enough or max iterations reached)
                is_good_enough = (segment_result['utility_score'] >= self.utility_threshold and 
                                 segment_result['support_score'] >= 1)
                is_last_iteration = iteration == self.max_iterations - 1
                
                # Log decision for each question
                logger.info(f"[SELFRAG] Question {i+1} evaluation:")
                logger.info(f"[SELFRAG] Utility score: {segment_result['utility_score']} (threshold: {self.utility_threshold})")
                logger.info(f"[SELFRAG] Support score: {segment_result['support_score']} (threshold: 1)")
                logger.info(f"[SELFRAG] Needs retrieval: {segment_result['needs_retrieval']}")
                logger.info(f"[SELFRAG] Is good enough: {is_good_enough}")
                logger.info(f"[SELFRAG] Is last iteration: {is_last_iteration}")
                
                if is_good_enough or is_last_iteration or not segment_result['needs_retrieval']:
                    # Question is complete
                    logger.info(f"[SELFRAG] Question {i+1} COMPLETED after {iteration + 1} iterations")
                    final_results[orig_idx] = {
                        'question': active_questions[i],
                        'final_context': active_contexts[i],
                        'final_answer': segment_result['final_answer'],
                        'final_support_score': segment_result['support_score'],
                        'final_utility_score': segment_result['utility_score'],
                        'iterations_used': iteration + 1,
                        'segments': segment_result['segments']
                    }
                else:
                    # Question needs more retrieval
                    logger.info(f"[SELFRAG] Question {i+1} needs MORE RETRIEVAL")
                    questions_needing_retrieval.append(active_questions[i])
                    contexts_needing_retrieval.append(active_contexts[i])
                    indices_needing_retrieval.append(i)
                    
                    new_active_questions.append(active_questions[i])
                    new_active_contexts.append(active_contexts[i])
                    new_active_indices.append(orig_idx)
                
                logger.info("---")
            
            # Batch retrieve additional documents for questions that need it
            if questions_needing_retrieval and iteration < self.max_iterations - 1:
                logger.info(f"[SELFRAG] Retrieving additional docs for {len(questions_needing_retrieval)} questions")
                additional_docs = database.batch_search(questions_needing_retrieval, 
                                                      list(range(len(questions_needing_retrieval))), k=15)
                relevant_additional_docs = self.filter_relevant_documents(questions_needing_retrieval, additional_docs)
                
                # Update contexts with additional information
                for i, relevant_docs in enumerate(relevant_additional_docs):
                    if relevant_docs:
                        context_idx = indices_needing_retrieval[i]
                        additional_context = '\n- '.join(relevant_docs[:2])
                        new_active_contexts[context_idx] += "\n\nAdditional information:\n" + additional_context
                        logger.info(f"[SELFRAG] Added {len(relevant_docs[:2])} additional docs to question {i+1}")
                    else:
                        logger.info(f"[SELFRAG] No additional relevant docs found for question {i+1}")
            elif questions_needing_retrieval:
                logger.info(f"[SELFRAG] Skipping additional retrieval (last iteration or no questions needing retrieval)")
            
            # Update active lists for next iteration
            active_questions = new_active_questions
            active_contexts = new_active_contexts
            active_indices = new_active_indices
        
        return final_results

    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        embedding_model = "all-MiniLM-L6-v2"
        sample = samples[0]

        # Step 1: Evaluate if retrieval is needed
        questions = [sample.get('question', '') for sample in samples]
        retrieval_needed = self.llm_system.evaluate_retrieval_need(questions)
        retrieval_needed_questions = [questions[i] for i in range(len(questions)) if retrieval_needed[i]]
        retrieval_needed_samples = [(i, samples[i]) for i in range(len(samples)) if retrieval_needed[i]]
        non_retrieval_questions = [questions[i] for i in range(len(questions)) if not retrieval_needed[i]]
        non_retrieval_samples = [(i, samples[i]) for i in range(len(samples)) if not retrieval_needed[i]]
        
        # Log initial retrieval decisions
        logger.info(f"[SELFRAG] === BATCH PROCESSING {len(samples)} SAMPLES ===")
        logger.info(f"[SELFRAG] Questions needing retrieval: {len(retrieval_needed_questions)}")
        logger.info(f"[SELFRAG] Questions NOT needing retrieval: {len(non_retrieval_questions)}")
        
        for i, (question, needs_retrieval) in enumerate(zip(questions, retrieval_needed)):
            logger.info(f"[SELFRAG] Sample {i+1}: {question[:100]}... -> {'RETRIEVE' if needs_retrieval else 'NO RETRIEVE'}")
        logger.info("---")

        # Setup database
        if sample.get('search_results', []) != [] and \
            sample['search_results'][0].get('persistent_storage', None):
            if not hasattr(self, 'database'):
                self.database = ChunkSearcher(embedding_model=embedding_model)
                self.database.set_documents([get_storage(sample['search_results'][0]['persistent_storage'])])
            database = self.database
        else:
            documents = [[doc['page_snippet'] + "\n\n" + clean_web_content(doc.get('page_result', ''))
                        for doc in _sample.get('search_results', [])] for _, _sample in retrieval_needed_samples]
            database = ChunkSearcher(embedding_model=embedding_model)
            database.set_documents(documents)

        augmented_samples = samples.copy()
        
        # Process samples that need retrieval with iterative refinement
        if retrieval_needed_questions:
            # Get initial context for retrieval-needed samples
            initial_retrieved_docs = database.batch_search(retrieval_needed_questions, [i for i in range(len(retrieval_needed_samples))], k=30)
            initial_relevant_docs = self.filter_relevant_documents(retrieval_needed_questions, initial_retrieved_docs)
            initial_contexts = ['\n- '.join(docs)[:3000] for docs in initial_relevant_docs]
            
            # Apply iterative retrieval and refinement
            refined_results = self.iterative_retrieve_and_refine(retrieval_needed_questions, database, initial_contexts)
            
            # Evaluate utility for all options in batch
            confirm_questions = []
            confirm_options = []
            confirm_contexts = []
            confirm_ids = []
            
            for (i, sample), refined_result in zip(retrieval_needed_samples, refined_results):
                options = sample.get('options', [])
                confirm_options += options
                confirm_questions += [sample.get('question', '')] * len(options)
                confirm_contexts += [refined_result['final_context']] * len(options)
                confirm_ids += [i] * len(options)
            
            # Batch evaluate support and utility
            confirm_supports = self.llm_system.evaluate_support(confirm_questions, confirm_contexts, confirm_options)
            confirm_utilities = self.llm_system.evaluate_utility(confirm_questions, confirm_contexts, confirm_options)
            
            # Update augmented samples with refined results
            for (i, sample), refined_result in zip(retrieval_needed_samples, refined_results):
                augmented_samples[i]['context'] = refined_result['final_context']
                augmented_samples[i]['support'] = refined_result['final_support_score']
                augmented_samples[i]['utility'] = refined_result['final_utility_score']
                augmented_samples[i]['iterations_used'] = refined_result['iterations_used']
                augmented_samples[i]['refined_answer'] = refined_result['final_answer']

        # Process non-retrieval samples with segment generation for consistency
        if non_retrieval_questions:
            non_retrieval_segment_results = self.llm_system.generate_with_segments(non_retrieval_questions, [""] * len(non_retrieval_questions), self.max_segments)
            
            for (i, sample), segment_result in zip(non_retrieval_samples, non_retrieval_segment_results):
                augmented_samples[i]['context'] = ""
                augmented_samples[i]['support'] = segment_result['support_score']
                augmented_samples[i]['utility'] = segment_result['utility_score']
                augmented_samples[i]['iterations_used'] = 1
                augmented_samples[i]['refined_answer'] = segment_result['final_answer']

        # Generate final responses
        final_results = []
        for i, augmented_sample in enumerate(augmented_samples):
            base_result = self.llm_system.process_sample(augmented_sample)
            final_results.append({
                **base_result,
                'retrieval_needed': retrieval_needed[i],
                'context': augmented_sample.get('context', ''),
                'support_score': augmented_sample.get('support', 0),
                'utility_score': augmented_sample.get('utility', 3),
                'iterations_used': augmented_sample.get('iterations_used', 1),
                'refined_answer': augmented_sample.get('refined_answer', base_result.get('generated_response', ''))
            })
        
        return final_results
