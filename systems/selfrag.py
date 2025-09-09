from systems.abstract import AbstractRAGSystem
from systems.selfllm import SelfLLMSystem
from typing import Dict, Any, List
from utils.ramdb import ChunkSearcher  # pyright: ignore[reportMissingImports]
from utils.clean import clean_web_content  # pyright: ignore[reportMissingImports]
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
        self.llm_system = SelfLLMSystem(model_name, device, technique='rag')
        self.max_iterations = kwargs.get('max_iterations', 3)
        self.relevance_threshold = kwargs.get('relevance_threshold', 0.7)
    
    def get_batch_size(self) -> int: return 1
    
    def filter_relevant_documents(self, questions: List[str], retrieved_docs: List[List[str]]) -> List[Dict[str, Any]]:
        """Filter documents based on relevance evaluation."""
        relevant_docs = [[] for _ in questions]
        is_relevant_batch = self.llm_system.evaluate_relevance(questions, retrieved_docs)
        
        for i, (is_relevant, docs) in enumerate(zip(is_relevant_batch, retrieved_docs)):
            for is_relevant_doc, doc in zip(is_relevant, docs):
                if is_relevant_doc:
                    relevant_docs[i].append(doc)
        
        return relevant_docs

    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        embedding_model = "all-MiniLM-L6-v2"
        sample = samples[0]

        # Step 1: Evaluate if retrieval is needed
        questions = [sample.get('question', '') for sample in samples]
        retrieval_needed = self.llm_system.evaluate_retrieval_need(questions)
        retrieval_needed_questions = [questions[i] for i in range(len(questions)) if retrieval_needed[i]]
        retrieval_needed_samples = [(i, samples[i]) for i in range(len(samples)) if retrieval_needed[i]]

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
        if retrieval_needed_questions:

            retrieved_docs = database.batch_search(retrieval_needed_questions, [i for i in range(len(retrieval_needed_samples))], k=50)

            relevant_docs = self.filter_relevant_documents(retrieval_needed_questions, retrieved_docs)
            relevant_docs = ['\n- '.join(docs)[:4000] for docs in relevant_docs]

            confirm_questions = []
            confirm_options = []
            confirm_contexts = []
            confirm_ids = []
            for (i, sample), relevant_doc in zip(retrieval_needed_samples, relevant_docs):
                options = sample.get('options', [])
                confirm_options += options
                confirm_questions += [sample.get('question', '')] * len(options)
                confirm_contexts += [relevant_doc] * len(options)
                confirm_ids += [i] * len(options)
            
            confirm_supports = self.llm_system.evaluate_support(confirm_questions, confirm_contexts, confirm_options)
            
            inform_format = '''
    After reading the context, the agent determines that it is {inform_text} with respect to the answer {option}.
    '''

            for id, context, question, support, option in zip(confirm_ids, confirm_contexts, confirm_questions, confirm_supports, confirm_options):
                if 'support' not in augmented_samples[id] or support >= augmented_samples[id]['support']:
                    augmented_samples[id]['support'] = support
                    if support == 2: inform_text = 'fully supported'
                    elif support == 1: inform_text = 'partially supported'
                    else: inform_text = 'not supported'
                    augmented_samples[id]['context'] = context + inform_format.format(inform_text=inform_text, option=option)

        return [{
            **self.llm_system.process_sample(augmented_sample),
            'retrieval_needed': retrieval_needed[i],
            'context': augmented_sample.get('context', ''),
        }
        for i, augmented_sample in enumerate(augmented_samples)]
