from systems.abstract import AbstractRAGSystem
from systems.hydellm import HyDELLMSystem
from typing import Dict, Any, List
import torch
from loguru import logger
from utils.clean import clean_web_content  # pyright: ignore[reportMissingImports]
from utils.storage import get_storage  # pyright: ignore[reportMissingImports]
from utils.ramdb import ChunkSearcher

class HyDERAGSystem(AbstractRAGSystem):
    """
    HyDE RAG system that uses hypothetical document embeddings for retrieval.
    
    This system implements the HyDE (Hypothetical Document Embeddings) technique:
    1. Generate a hypothetical document that would answer the query
    2. Use that hypothetical document for retrieval instead of the original query
    3. Retrieve relevant documents based on hypothetical document similarity
    4. Generate final answer using retrieved context
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", num_samples: int = 20, **kwargs):
        self.llm_system = HyDELLMSystem(model_name, device, num_samples=num_samples, technique='rag')
    
    def get_batch_size(self) -> int: return 50
    
    def generate_hypothetical_document(self, questions: List[str]) -> List[str]:
        system_message = "You are a helpful assistant that writes comprehensive and informative passages to answer questions."
        user_message = "Write a detailed, factual passage that would answer the following question:\n\nQuestion: {}\n\nProvide a comprehensive answer with relevant facts and information."

        prompts = [self.llm_system._create_unified_prompt(system_message, user_message.format(question)) for question in questions]

        inputs = self.llm_system.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.llm_system.device)
        
        with torch.no_grad():
            outputs = self.llm_system.model.generate(
                **inputs,
                max_new_tokens=224,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.llm_system.tokenizer.eos_token_id,
                eos_token_id=self.llm_system.tokenizer.eos_token_id
            )
        
        hypothetical_docs = [self.llm_system.tokenizer.decode(output[len(inputs.input_ids[i]):], skip_special_tokens=True).strip() for i, output in enumerate(outputs)]

        return hypothetical_docs

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

        hypothetical_docs = self.generate_hypothetical_document([sample.get('question', '') for sample in samples])
        retrieved_docs = database.batch_search(hypothetical_docs, [i for i in range(len(samples))], k=10)
        
        for _id, sample in enumerate(samples):
            retrieved_doc = retrieved_docs[_id]
            query_time = sample.get('query_time', 'March 1, 2025')
            augmented_sample = sample.copy()
            if retrieved_doc: augmented_sample['context'] = ("\n- " + "\n- ".join(retrieved_doc))[:4000] + '\nQuery Time: ' + query_time
            try:
                result = self.llm_system.process_sample(augmented_sample)
                results.append(result)
            except Exception as e:
                print(e)
        
        return results
        