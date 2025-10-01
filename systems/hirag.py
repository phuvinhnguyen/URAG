from systems.abstract import AbstractRAGSystem
from systems.hillm import HiLLMSystem
from typing import Dict, Any, List
import os
import logging
import numpy as np
import httpx
import asyncio
from HiRAG.hirag import HiRAG, QueryParam
from dataclasses import dataclass
from HiRAG.hirag.base import BaseKVStorage
from HiRAG.hirag._utils import compute_args_hash
from utils.clean import clean_web_content
from utils.storage import get_storage

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""
    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func
    return final_decro

def create_embedding_function(embedding_type: str, embedding_model: str, embedding_config: Dict, embedding_dim: int = 768, max_token_size: int = 8192):
    """Factory function to create embedding function with specific configuration"""
    
    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=max_token_size)
    async def embedding_function(texts: list[str]) -> np.ndarray:
        """Universal embedding function supporting multiple embedding types"""
        
        if embedding_type == "vllm":
            return await vllm_embedding(texts, embedding_config, embedding_model)
        elif embedding_type == "sentence_transformers":
            return await sentence_transformers_embedding(texts, embedding_model)
        elif embedding_type == "openai_compatible":
            return await openai_compatible_embedding(texts, embedding_config, embedding_model)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    return embedding_function

async def vllm_embedding(texts: list[str], embedding_config: Dict, embedding_model: str) -> np.ndarray:
    """vLLM embedding function using HTTP requests to vLLM embedding server"""
    embedding_base_url = embedding_config['base_url']
    embedding_api_key = embedding_config['api_key']
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{embedding_base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {embedding_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": embedding_model,
                    "input": texts,
                    "encoding_format": "float"
                }
            )
            response.raise_for_status()
            data = response.json()
            return np.array([item["embedding"] for item in data["data"]])
        except Exception as e:
            logging.error(f"vLLM embedding failed: {e}")
            raise

async def sentence_transformers_embedding(texts: list[str], embedding_model: str) -> np.ndarray:
    """Local sentence-transformers embedding function"""
    try:
        from sentence_transformers import SentenceTransformer
        import asyncio
        
        # Load model (you might want to cache this globally)
        logging.info(f"Loading sentence-transformers model: {embedding_model}")
        model = SentenceTransformer(embedding_model)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, model.encode, texts)
        
        embeddings_array = np.array(embeddings)
        logging.info(f"Generated embeddings shape: {embeddings_array.shape} for {len(texts)} texts using model {embedding_model}")
        
        return embeddings_array
    except ImportError:
        raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
    except Exception as e:
        logging.error(f"Sentence transformers embedding failed: {e}")
        raise

async def openai_compatible_embedding(texts: list[str], embedding_config: Dict, embedding_model: str) -> np.ndarray:
    """OpenAI-compatible API embedding function"""
    embedding_base_url = embedding_config['base_url']
    embedding_api_key = embedding_config['api_key']
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{embedding_base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {embedding_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": embedding_model,
                    "input": texts,
                    "encoding_format": "float"
                }
            )
            response.raise_for_status()
            data = response.json()
            return np.array([item["embedding"] for item in data["data"]])
        except Exception as e:
            logging.error(f"OpenAI-compatible embedding failed: {e}")
            raise

def create_vllm_model_function(vllm_model: str, vllm_base_url: str, vllm_api_key: str):
    """Factory function to create vLLM model function with specific configuration"""
    
    async def VLLM_model_if_cache(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        """vLLM model function using HTTP requests to vLLM server"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Get the cached response if having-------------------
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        if hashing_kv is not None:
            args_hash = compute_args_hash(vllm_model, messages)
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]
        # -----------------------------------------------------

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                payload = {
                    "model": vllm_model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "stream": False
                }
                
                response = await client.post(
                    f"{vllm_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {vllm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Cache the response if having-------------------
                if hashing_kv is not None:
                    await hashing_kv.upsert(
                        {args_hash: {"return": content, "model": vllm_model}}
                    )
                # -----------------------------------------------------
                
                return content
                
            except Exception as e:
                logging.error(f"vLLM API call failed: {e}")
                raise
    
    return VLLM_model_if_cache

class HiRAGSystem(AbstractRAGSystem):    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 technique: str = 'rag',
                 working_dir: str = "test_hirag",
                 enable_llm_cache: bool = False,
                 enable_hierachical_mode: bool = True,
                 embedding_batch_num: int = 6,
                 embedding_func_max_async: int = 8,
                 enable_naive_rag: bool = True,
                 # vLLM configuration
                 vllm_host: str = "localhost",
                 vllm_port: int = 8000,
                 vllm_base_url: str = None,
                 vllm_api_key: str = "123",
                 # Embedding configuration
                 embedding_type: str = "sentence_transformers",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_dim: int = 384,
                 embedding_host: str = "localhost",
                 embedding_port: int = 8001,
                 embedding_base_url: str = None,
                 embedding_api_key: str = "dummy",
                 max_token_size: int = 8192,
                 **kwargs):
        
        # Initialize LLM system with vLLM configuration
        self.llm_system = HiLLMSystem(
            model_name=model_name,
            technique=technique,
            vllm_host=vllm_host,
            vllm_port=vllm_port,
            vllm_base_url=vllm_base_url,
            vllm_api_key=vllm_api_key,
            **kwargs
        )
        
        # Configure embedding
        # Handle case where embedding_host already includes full URL
        if embedding_base_url:
            final_embedding_base_url = embedding_base_url
        elif embedding_host.startswith('http'):
            final_embedding_base_url = embedding_host
        else:
            final_embedding_base_url = f"http://{embedding_host}:{embedding_port}/v1"
        
        embedding_config = {
            'base_url': final_embedding_base_url,
            'api_key': embedding_api_key
        }
        
        # Create embedding function
        embedding_function = create_embedding_function(
            embedding_type=embedding_type,
            embedding_model=embedding_model,
            embedding_config=embedding_config,
            embedding_dim=embedding_dim,
            max_token_size=max_token_size
        )
        
        # Create vLLM model function
        # Handle case where vllm_host already includes full URL
        if vllm_host.startswith('http'):
            vllm_base_url_final = vllm_host
        else:
            vllm_base_url_final = vllm_base_url or f"http://{vllm_host}:{vllm_port}/v1"
        
        vllm_model_function = create_vllm_model_function(
            vllm_model=model_name,
            vllm_base_url=vllm_base_url_final,
            vllm_api_key=vllm_api_key
        )
        
        # Debug logging for configuration
        logging.info(f"HiRAG initialized with vLLM URL: {vllm_base_url_final}")
        logging.info(f"HiRAG initialized with embedding URL: {final_embedding_base_url}")
        logging.info(f"HiRAG embedding config: type={embedding_type}, model={embedding_model}, dim={embedding_dim}")
        
        # Initialize HiRAG with configuration
        self.hirag = HiRAG(
            working_dir=working_dir,
            enable_llm_cache=enable_llm_cache,
            embedding_func=embedding_function,
            best_model_func=vllm_model_function,
            cheap_model_func=vllm_model_function,
            enable_hierachical_mode=enable_hierachical_mode, 
            embedding_batch_num=embedding_batch_num,
            embedding_func_max_async=embedding_func_max_async,
            enable_naive_rag=enable_naive_rag
        )
        
        self._documents_indexed = False
    
    def get_batch_size(self) -> int: 
        return 10

    def _index_documents_from_samples(self, samples: List[Dict[str, Any]]):
        """Index documents from samples' page_result content using HiRAG"""
        if self._documents_indexed:
            return
            
        # Extract all page_result content for indexing
        documents_to_index = []
        
        for sample in samples:
            search_results = sample.get('search_results', [])
            for result in search_results:
                # Check if there's persistent storage first
                if result.get('persistent_storage', None):
                    # Load from persistent storage
                    try:
                        stored_content = get_storage(result['persistent_storage'])
                        documents_to_index.append(stored_content)
                    except Exception as e:
                        logging.warning(f"Could not load from persistent storage: {e}")
                        # Fallback to page_result
                        page_result = result.get('page_result', '')
                        if page_result:
                            cleaned_content = clean_web_content(page_result)
                            documents_to_index.append(cleaned_content)
                else:
                    # Use page_result directly
                    page_result = result.get('page_result', '')
                    if page_result:
                        cleaned_content = clean_web_content(page_result)
                        documents_to_index.append(cleaned_content)
        
        # Index each document separately in HiRAG for better performance
        if documents_to_index:
            try:
                # Index each document separately for better semantic understanding
                for i, document in enumerate(documents_to_index):
                    logging.info(f"Indexing document {i+1}/{len(documents_to_index)} (length: {len(document)} chars)")
                    # Use synchronous insert method to avoid event loop conflicts
                    try:
                        self.hirag.insert(document)
                    except ValueError as ve:
                        if "dimensions except for the concatenation axis must match exactly" in str(ve):
                            logging.error(f"Embedding dimension mismatch detected: {ve}")
                            logging.error("This usually happens when the vector database was created with different embedding dimensions.")
                            logging.error("Please delete the working directory or change the working_dir to start fresh.")
                            raise ValueError("Embedding dimension mismatch. Please delete the working directory and restart.") from ve
                        else:
                            raise
                    except IndexError as ie:
                        if "index 0 is out of bounds for axis 0 with size 0" in str(ie):
                            logging.error(f"Vector database corruption detected: {ie}")
                            logging.error("This usually happens when the vector database is corrupted or has dimension mismatches.")
                            logging.error("Please delete the working directory or change the working_dir to start fresh.")
                            raise ValueError("Vector database corruption. Please delete the working directory and restart.") from ie
                        else:
                            raise
                
                self._documents_indexed = True
                logging.info(f"Successfully indexed {len(documents_to_index)} documents separately in HiRAG")
                
            except Exception as e:
                logging.error(f"Error indexing documents: {e}")
                raise

    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        
        # Index documents from all samples first
        self._index_documents_from_samples(samples)
        
        for sample in samples:
            try:
                query_time = sample.get('query_time', 'March 1, 2025')
                question = sample.get('question', '')
                
                # Use HiRAG to retrieve relevant context
                # Query HiRAG with hierarchical search - use synchronous method to avoid event loop conflicts
                hirag_result = self.hirag.query(question, param=QueryParam(mode="hi"))
                
                # Create augmented sample with HiRAG context
                augmented_sample = sample.copy()
                if hirag_result:
                    # Limit context to avoid token overflow
                    context = str(hirag_result)[:4000] + f'\nQuery Time: {query_time}'
                    augmented_sample['context'] = context
                else:
                    augmented_sample['context'] = f'Query Time: {query_time}'
                
                # Process with the LLM system
                result = self.llm_system.process_sample(augmented_sample)
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                # Fallback to processing without RAG context
                try:
                    result = self.llm_system.process_sample(sample)
                    results.append(result)
                except Exception as fallback_e:
                    logging.error(f"Fallback processing also failed: {fallback_e}")
        
        return results
