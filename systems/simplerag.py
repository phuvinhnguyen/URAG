from systems.abstract import AbstractRAGSystem
from systems.simplellm import SimpleLLMSystem
from typing import Dict, Any
from loguru import logger
from utils.clean import clean_web_content
from utils.ramdb import ChunkSearcher
from utils.get_html import get_web_content
from utils.storage import get_storage

class SimpleRAGSystem(AbstractRAGSystem):    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", device: str = "cuda", **kwargs):
        self.llm_system = SimpleLLMSystem(model_name, device, technique='rag')
    
    def get_batch_size(self) -> int: return 1
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        config = { "embedding_model": "all-MiniLM-L6-v2" }
        num_retrieved_docs = 10

        question = sample.get('question', '')
        if sample.get('search_results', []) != [] and \
            sample['search_results'][0].get('persistent_storage', None):
            if not hasattr(self, 'database'):
                self.database = ChunkSearcher(embedding_model=config['embedding_model'])
                self.database.set_documents([get_storage(sample['search_results'][0]['persistent_storage'])])
            database = self.database
        else:
            documents = [doc['page_snippet'] + "\n\n" + clean_web_content(doc['page_result']) if doc['page_result'] else
                        doc['page_snippet'] + "\n\n" + clean_web_content(get_web_content(doc['page_url']))
                        for doc in sample.get('search_results', [])]

            database = ChunkSearcher(embedding_model=config['embedding_model'])
            database.set_documents([documents])

        retrieved_docs = database.search(question, k=num_retrieved_docs)

        # clean and remove database from memory to save RAM as much as possible
        del database
        
        # Augment sample with retrieved context
        augmented_sample = sample.copy()
        if retrieved_docs: augmented_sample['context'] = "- " + "\n- ".join(retrieved_docs)
        
        # Process through LLM
        result = self.llm_system.process_sample(augmented_sample)
        
        return result
