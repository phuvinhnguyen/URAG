from systems.abstract import AbstractRAGSystem
from systems.simplellm import SYSTEM_PROMPT
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List
from pathlib import Path
import os
from utils.clean import clean_web_content
from utils.ramdb import ChunkSearcher
from utils.storage import get_storage

from systems.raptorllm import LocalQAModel, LocalSummarizationModel, LocalEmbeddingModel

try:
    from raptor.raptor import RetrievalAugmentation, RetrievalAugmentationConfig
    RAPTOR_AVAILABLE = True
except ImportError:
    # cd to parrent dirrectory of this file and clone https://github.com/parthsarthi03/raptor.git
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.system(f"cd {current_dir}/../ && git clone https://github.com/parthsarthi03/raptor.git")

    try:
        from raptor.raptor import RetrievalAugmentation, RetrievalAugmentationConfig
        RAPTOR_AVAILABLE = True
    except ImportError:
        RAPTOR_AVAILABLE = False


class RaptorRAGSystem(AbstractRAGSystem):
    """
    Simplified RAPTOR RAG system following SimpleRAGSystem patterns.
    Inherits from AbstractRAGSystem and combines LLM + RAPTOR + RAG functionality.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda", 
                 technique: str = "rag", max_new_tokens: int = 100, temperature: float = 0.1, 
                 method: str = 'normal', embedding_model: str = "all-MiniLM-L6-v2", **kwargs):
        
        # Initialize LLM (following RaptorLLMV2System pattern)
        self.device = device
        self.technique = technique
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.method = method
        self.embedding_model = embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if RAPTOR_AVAILABLE:
            # Create RAPTOR models using the shared LLM
            self.qa_model = LocalQAModel(self.model, self.tokenizer)
            self.summarization_model = LocalSummarizationModel(self.model, self.tokenizer)
            self.embedding_model = LocalEmbeddingModel(model_name=embedding_model)
            
            # Initialize RAPTOR configuration
            self.config = RetrievalAugmentationConfig(
                qa_model=self.qa_model,
                summarization_model=self.summarization_model,
                embedding_model=self.embedding_model
            )
            
            # Create RAPTOR system (no tree persistence)
            self.raptor = RetrievalAugmentation(config=self.config)
            self.tree_built = False
        else:
            self.raptor = None
    
    def get_batch_size(self) -> int: return 40
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt based on sample technique using SYSTEM_PROMPT."""
        question = sample.get('question', '')
        
        if self.technique == 'cot': 
            prompt = f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        elif self.technique == 'rag': 
            prompt = f"Context information: {sample.get('context', '')}\n\nQuestion: {question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        else: 
            prompt = f"{question}\n\nPlease provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."
        
        try:
            prompt = self.tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
                ], tokenize=False, add_generation_prompt=True)
        except: 
            pass

        return prompt
    
    def _build_tree_if_needed(self, sample: Dict[str, Any]):
        """Build RAPTOR tree from context if available and not already built."""
        if not RAPTOR_AVAILABLE or self.tree_built:
            return
        
        context = sample.get('search_results', sample.get('context', ''))
        if context and isinstance(context, str) and len(context.strip()) > 50:
            try:
                self.raptor.add_documents(context)
                self.tree_built = True
            except Exception:
                pass  # Fallback to normal LLM processing
    
    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Efficient batch processing with single database initialization."""
        results = []
        embedding_model = self.embedding_model
        
        # Check for persistent storage (like SimpleRAGSystem)
        sample = samples[0]
        if sample.get('search_results', []) != [] and \
            isinstance(sample['search_results'], list) and \
            len(sample['search_results']) > 0 and \
            isinstance(sample['search_results'][0], dict) and \
            sample['search_results'][0].get('persistent_storage', None):
            
            # Use persistent storage - reuse database if already created
            if not hasattr(self, 'database'):
                self.database = ChunkSearcher(embedding_model=embedding_model)
                self.database.set_documents([get_storage(sample['search_results'][0]['persistent_storage'])])
            database = self.database
        else:
            # Create documents from search results (like SimpleRAGSystem)
            documents = [[doc.get('page_snippet', '') + "\n\n" + clean_web_content(doc.get('page_result', ''))
                        for doc in _sample.get('search_results', [])] for _sample in samples]
            database = ChunkSearcher(embedding_model=embedding_model)
            database.set_documents(documents)
        
        # Batch search for all questions at once
        retrieved_docs = database.batch_search(
            [sample.get('question', '') for sample in samples],
            [i for i in range(len(samples))],
            k=10)
        
        # Process each sample with retrieved context
        for sample, retrieved_doc in zip(samples, retrieved_docs):
            query_time = sample.get('query_time', 'March 1, 2025')
            augmented_sample = sample.copy()
            
            # Add retrieved context (like SimpleRAGSystem)
            if retrieved_doc:
                augmented_sample['context'] = ("\n- " + "\n- ".join(retrieved_doc))[:4000] + '\nQuery Time: ' + query_time
            
            try:
                # Process with RAPTOR + RAG capabilities
                result = self._process_single_sample(augmented_sample)
                results.append(result)
            except Exception as e:
                print(f"Error processing sample: {e}")
                # Add error result
                error_result = {
                    'id': sample.get('id', 'unknown'),
                    'generated_response': f"Error: {str(e)}",
                    'predicted_answer': '',
                    'conformal_probabilities': {},
                    'technique': 'rag',
                    'raptor_used': False,
                    'error': str(e)
                }
                results.append(error_result)
        
        return results
    
    def _process_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample with RAPTOR + RAG capabilities."""
        # Try to build tree from context
        self._build_tree_if_needed(sample)
        
        raptor_used = False
        enhanced_sample = sample.copy()
        
        # Try RAPTOR question answering first
        if RAPTOR_AVAILABLE and self.raptor and hasattr(self.raptor, 'tree') and self.raptor.tree:
            try:
                question = sample.get('question', '')
                raptor_answer = self.raptor.answer_question(question=question)
                
                if raptor_answer and len(raptor_answer.strip()) > 0:
                    # Combine RAPTOR answer with existing context
                    existing_context = enhanced_sample.get('context', '')
                    if existing_context:
                        enhanced_sample['context'] = f"{existing_context}\n\nRAPTOR Analysis: {raptor_answer}"
                    else:
                        enhanced_sample['context'] = f"RAPTOR Analysis: {raptor_answer}"
                    raptor_used = True
            except Exception:
                pass  # Continue with regular processing
        
        # Generate response using AbstractRAGSystem methods with SYSTEM_PROMPT
        prompt = self._generate_prompt(enhanced_sample)
        options = enhanced_sample.get('options', [])
        response, conformal_probabilities = self._generate_response_with_probabilities(prompt, options)
        
        # Get predicted answer safely
        predicted_answer = ""
        if conformal_probabilities:
            predicted_answer = max(conformal_probabilities.items(), key=lambda x: x[1])[0]
        
        return {
            'id': enhanced_sample.get('id', 'unknown'),
            'generated_response': response,
            'predicted_answer': predicted_answer,
            'conformal_probabilities': conformal_probabilities,
            'technique': self.technique,
            'raptor_used': raptor_used,
            'rag_enhanced': bool(sample.get('context', ''))
        }
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Single sample processing - delegates to batch for consistency."""
        results = self.batch_process_samples([sample])
        return results[0] if results else {
            'id': sample.get('id', 'unknown'),
            'generated_response': 'Error: No result from batch processing',
            'predicted_answer': '',
            'conformal_probabilities': {},
            'technique': 'rag',
            'raptor_used': False,
            'error': 'Batch processing failed'
        }
