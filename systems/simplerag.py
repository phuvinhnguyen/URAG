from systems.abstract import AbstractRAGSystem
from systems.simplellm import SimpleLLMSystem
from typing import Dict, Any, List
from utils.clean import clean_web_content
from utils.ramdb import ChunkSearcher
from utils.get_html import get_web_content
from utils.storage import get_storage

class SimpleRAGSystem(AbstractRAGSystem):    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", device: str = "cuda", **kwargs):
        self.llm_system = SimpleLLMSystem(model_name, device, technique='rag')
    
    def get_batch_size(self) -> int: return 20

    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        embedding_model = "all-MiniLM-L6-v2"
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
        
        for _id, sample in enumerate(samples):
            retrieved_docs = database.search(sample.get('question', ''), interaction_id=_id, k=10)
            query_time = sample.get('query_time', 'March 1, 2025')
            augmented_sample = sample.copy()
            if retrieved_docs: augmented_sample['context'] = ("\n- " + "\n- ".join(retrieved_docs))[:4000] + '\nQuery Time: ' + query_time
            try:
                result = self.llm_system.process_sample(augmented_sample)
                results.append(result)
            except Exception as e:
                print(e)        
        
        return results