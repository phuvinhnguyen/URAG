from systems.abstract import AbstractRAGSystem
from systems.ratllm import RATLLMSystem, SYSTEM_PROMPT
from typing import Dict, Any, List
from utils.clean import clean_web_content
from utils.ramdb import ChunkSearcher
from utils.storage import get_storage
import re

RAT_QUERY_GENERATION_PROMPT = """You are generating search queries to find evidence for reasoning steps. Create a focused search query that will help verify or improve the current reasoning.

Instructions:
1. Analyze the question and current reasoning
2. Identify what additional evidence would be most helpful
3. Generate a specific, focused search query
4. Keep the query concise but informative

Example:
Question: What is the capital of France?
Current reasoning: France is a European country with major cities.
Search query: capital city of France official government seat
"""

RAT_REVISION_PROMPT = """You are revising reasoning step-by-step using retrieved evidence. Improve each reasoning step with the provided context while maintaining logical flow.

Instructions:
1. Use the evidence to enhance your reasoning
2. Keep reasoning steps clear and logical
3. Build upon previous steps progressively
4. End with Answer|X where X is the correct option (A, B, C, D, etc.)

Example:
Question: What is the capital of France?
Evidence: Paris is the capital and largest city of France.
Reasoning: 1. France is a country in Europe. 2. Based on the evidence, Paris is the capital city of France. 3. Therefore, the answer is the option that states Paris.
Answer|B
"""

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
                 max_iterations: int = 1, retrieval_k: int = 10, max_chunks_per_thought: int = 2, thought_limit: int = 3, **kwargs):
        self.max_iterations = max_iterations  # iterations per thought
        self.retrieval_k = retrieval_k
        self.max_chunks_per_thought = retrieval_k
        self.thought_limit = thought_limit
        self.llm_system = RATLLMSystem(
            model_name=model_name, 
            device=device, 
            technique='rat',
            temperature=kwargs.pop('temperature', 0.1),  # Lower temperature for more focused revision
            **kwargs
        )
        self.tokenizer = self.llm_system.tokenizer
        self.device = self.llm_system.device
    
    def get_batch_size(self) -> int: return 2

    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample.get('question', '')
        system_message = SYSTEM_PROMPT + f""" You will provide step-by-step reasoning (exactly {self.thought_limit} reasoning steps in the first {self.thought_limit} lines) to answer the multiple choice question. Each reasoning step should be on a separate line. After all reasoning steps, provide short explanation before your final answer. You must strictly follow this format for the system to work."""
        user_message = f"Question: {question}\n\nProvide exactly {self.thought_limit} reasoning steps, then give your final answer in the format Answer|X where X is your chosen option (A, B, C, D, etc.).\n\nStep-by-step reasoning:"
        return self.llm_system._create_unified_prompt(system_message, user_message)

    def _generate_initial_thoughts_batch(self, questions: List[str], options: List[List[str]]) -> List[List[str]]:
        batch_cot_samples = [{
            'question': question,
            'options': option_list
        } for question, option_list in zip(questions, options)]
        batch_prompts = [self._generate_prompt(sample) for sample in batch_cot_samples] # prompts have special tokens
        batch_options = [sample.get('options', []) for sample in batch_cot_samples]
        batch_responses, _ = self.llm_system._generate_response_with_probabilities(batch_prompts, batch_options) # responses dont have special tokens
        # make sure prompts do not have special tokens by encode and decode again
        batch_prompts = [self.tokenizer.decode(self.tokenizer.encode(prompt, add_special_tokens=False), skip_special_tokens=True) for prompt in batch_prompts]
        batch_thought = [response.replace(prompt, '').strip() for response, prompt in zip(batch_responses, batch_prompts)]
        batch_thought = [[thought for thought in thoughts.split('\n') if thought.strip() != ''] for thoughts in batch_thought]
        # Concat the first thought with answer (after all thoughts)
        batch_thought = [
            [thought + '\n'.join(thoughts[self.thought_limit:]) if index == 0
            else thought
            for index, thought in enumerate(thoughts[:-1])
            ] + ['']*self.thought_limit for thoughts in batch_thought
            ]
        
        return batch_thought

    def _batch_generate_query_and_revise(self, batch_questions: List[str], batch_options: List[List[str]], 
                                         batch_current_reasoning: List[str], batch_corpus: List[str] = None) -> tuple[List[str], List[str]]:
        """Generate search queries and revise reasoning in a single LLM call for efficiency."""
        system_message = SYSTEM_PROMPT + " " + RAT_QUERY_GENERATION_PROMPT if not batch_corpus else SYSTEM_PROMPT + " " + RAT_REVISION_PROMPT
        
        if not batch_corpus:
            # Query generation phase
            user_message = """Current reasoning so far:
{current_reasoning}

Question: {question}

Generate a focused search query to find evidence that will help verify, improve this reasoning and answer the question (main purpose). Provide only the search query without any other text, symbols or anything else."""
            
            batch_prompts = [self.llm_system._create_unified_prompt(system_message, 
                            user_message.format(question=question, current_reasoning=reasoning)) 
                            for question, reasoning in zip(batch_questions, batch_current_reasoning)]
            batch_responses, _ = self.llm_system._generate_response_with_probabilities(batch_prompts, batch_options) # responses dont have special tokens while prompts do
            batch_prompts = [self.tokenizer.decode(self.tokenizer.encode(prompt, add_special_tokens=False), skip_special_tokens=True) for prompt in batch_prompts]
            batch_queries = [response.replace(prompt, '').strip() for response, prompt in zip(batch_responses, batch_prompts)]
            return batch_queries, batch_current_reasoning, None
        else:
            # Revision phase with evidence
            user_message = """Evidence found:
{evidence}

Current reasoning:
{current_reasoning}

Question: {question}

Based on the evidence and current reasoning, revise and provide step-by-step reasoning to answer the question. Maintain the step-by-step structure and end with Answer|X where X is your chosen option. Your answer MUST ends with Answer|X where X is your chosen option (A, B, C, D, etc.) to ensure the system works properly."""
            
            batch_prompts = [self.llm_system._create_unified_prompt(system_message,
                            user_message.format(question=question, current_reasoning=reasoning, evidence=corpus))
                            for question, reasoning, corpus in zip(batch_questions, batch_current_reasoning, batch_corpus)]
            batch_responses, conformal_probabilities = self.llm_system._generate_response_with_probabilities(batch_prompts, batch_options)
            batch_prompts = [self.tokenizer.decode(self.tokenizer.encode(prompt, add_special_tokens=False), skip_special_tokens=True) for prompt in batch_prompts]
            batch_revised_reasoning = [response.replace(prompt, '').strip() for response, prompt in zip(batch_responses, batch_prompts)]
            return [], batch_revised_reasoning, conformal_probabilities

    def _optimize_chunks(self, batch_chunks: List[List[str]]) -> List[str]:
        """Optimize chunk selection and processing for better context utilization."""
        optimized_corpus = []
        for chunks in batch_chunks:
            if not chunks:
                optimized_corpus.append("")
                continue
            max_length = 4000  # Leave room for prompt
            combined_text = '\n'.join([chunk[:1000] for chunk in chunks])[:max_length]            
            optimized_corpus.append(combined_text.strip())
        return optimized_corpus

    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimized RAT (Retrieval Augmented Thoughts) algorithm for batch processing.
        
        Algorithm 1: Retrieval Augmented Thoughts (RAT) - Optimized
        1. Generate zero-shot initial step-by-step thoughts: T = {T1, T2, ..., Tn}
        2. For each thought step, batch all operations:
           a. Generate all queries for current reasoning state
           b. Batch retrieve all information from corpus
           c. Revise all reasoning with retrieved evidence
        3. Return final reasoning with Answer|X format
        """
        # Setup database
        if samples[0].get('search_results', []) != [] and \
            samples[0]['search_results'][0].get('persistent_storage', None):
            if not hasattr(self, 'database'):
                self.database = ChunkSearcher()
                self.database.set_documents([get_storage(samples[0]['search_results'][0]['persistent_storage'])])
            database = self.database
        else:
            documents = [[doc['page_snippet'] + "\n\n" + clean_web_content(doc.get('page_result', ''))
                        for doc in _sample.get('search_results', [])] for _sample in samples]
            database = ChunkSearcher()
            database.set_documents(documents)

        batch_questions = [sample.get('question', '') for sample in samples]
        batch_options = [sample.get('options', ['A', 'B', 'C', 'D']) for sample in samples]
        batch_query_times = [sample.get('query_time', 'March 1, 2025') for sample in samples]

        # Step 1: Generate initial thoughts (step-by-step reasoning)
        batch_thoughts = self._generate_initial_thoughts_batch(batch_questions, batch_options)
        batch_current_reasoning = [thoughts[0] for thoughts in batch_thoughts]

        # Step 2: Iteratively refine reasoning with evidence (optimized batching)
        for index in range(self.thought_limit):
            # Generate all search queries for current reasoning state
            batch_queries, _, _ = self._batch_generate_query_and_revise(
                batch_questions, batch_options, batch_current_reasoning, None)
            # Batch search all queries at once
            query_indices = list(range(len(batch_questions)))
            batch_chunks = database.batch_search(batch_queries, query_indices, k=self.retrieval_k)
            
            # Optimize chunk processing
            batch_corpus = self._optimize_chunks(batch_chunks)
            
            # Revise reasoning with evidence
            _, batch_reasoning, _ = self._batch_generate_query_and_revise(
                batch_questions, batch_options, batch_current_reasoning, batch_corpus)

            if index >= self.thought_limit-1:
                batch_current_reasoning = [f'{j}\nReasoning: {i}' for i,j in zip(batch_reasoning, batch_corpus)]
                break

            batch_current_reasoning = [reasoning + '\n' + thoughts[index+1] for reasoning, thoughts in zip(batch_reasoning, batch_thoughts)]

        batch_samples = []
        for question, options, final_reasoning, query_time in zip(
            batch_questions, batch_options, batch_current_reasoning, batch_query_times):
            
            batch_samples.append({
                'question': question,
                'options': options,
                'context': f"Step-by-step reasoning with evidence:\n{final_reasoning}\nQuery Time: {query_time}"
            })
        
        return self.llm_system.batch_process_samples(batch_samples)