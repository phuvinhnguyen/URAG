from systems.abstract import AbstractRAGSystem
from systems.graphllm import GraphLLMSystem
from typing import Dict, Any, List, Optional
from loguru import logger
import pandas as pd
import os
import sys
import asyncio
import tempfile
import tiktoken
import requests
import json
from pathlib import Path

# Add GraphRAG to path for compatibility
graphrag_path = os.path.join(os.path.dirname(__file__), '..', 'graphrag')
sys.path.insert(0, graphrag_path)
sys.path.insert(0, os.path.join(graphrag_path, 'graphrag'))

try:
    from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
    from graphrag.query.indexer_adapters import (
        read_indexer_covariates,
        read_indexer_entities,
        read_indexer_relationships,
        read_indexer_reports,
        read_indexer_text_units,
    )
    from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
    from graphrag.query.structured_search.local_search.search import LocalSearch
    from graphrag.vector_stores.lancedb import LanceDBVectorStore
    from graphrag.api.index import build_index
    from graphrag.config.create_graphrag_config import create_graphrag_config
    from graphrag.config.enums import IndexingMethod
    GRAPHRAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GraphRAG not available: {e}")
    GRAPHRAG_AVAILABLE = False


class GraphRAGSystem(AbstractRAGSystem):
    """
    GraphRAG v2 system implementing Microsoft GraphRAG local search approach.
    
    This system follows the Microsoft GraphRAG local search example with:
    1. Question-specific context text embedding and indexing
    2. Local search using mixed context (entities, relationships, reports, text units)
    3. Targeted retrieval based on question context rather than global indexing
    4. Integration with SimpleLLMSystem for final response generation
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
                 device: str = "cuda",
                 # GraphRAG parameters
                 community_level: int = 2,
                 max_tokens: int = 12000,
                 # OpenAI-compatible API parameters
                 base_url: str = None,
                 api_key: str = None,
                 # GraphRAG data path (pre-indexed data)
                 graphrag_data_path: str = None,
                 # Auto-indexing parameters
                 auto_index: bool = True,
                 **kwargs):
        """Initialize GraphRAG v2 system for question-specific context processing."""
        
        if not GRAPHRAG_AVAILABLE:
            raise ImportError("GraphRAG is required but not available")
        
        # Store API parameters for LLM calls
        self.base_url = base_url
        self.api_key = api_key or "dummy-key"
        self.use_api = bool(base_url)
        self.model_name = model_name
        
        # GraphRAG configuration
        self.community_level = community_level
        self.max_tokens = max_tokens
        self.graphrag_data_path = graphrag_data_path
        
        # Use cl100k_base tokenizer for compatibility
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        
        # Context builder parameters following the provided example
        self.local_context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "max_tokens": self.max_tokens,
        }
        
        self.model_params = {
            "max_tokens": 2000,
            "temperature": 0.0,
        }
        
        # Storage for GraphRAG components
        self.entities = None
        self.relationships = None
        self.reports = None  
        self.text_units = None
        self.covariates = None
        self.description_embedding_store = None
        self.text_embedder = None
        self.chat_model = None
        
        # Auto-indexing state
        self.auto_index = auto_index
        self.indexed = False
        self.indexed_data_path = None
        
        # Initialize language models for API calls (needed regardless of GraphRAG data)
        self._initialize_language_models()
        
        # Load pre-indexed data if path provided
        if self.graphrag_data_path and os.path.exists(self.graphrag_data_path):
            self._load_graphrag_data()
        else:
            # Initialize empty data structures for context-only processing
            self.entities = []
            self.relationships = []
            self.reports = []
            self.text_units = []
            self.covariates = None
            self.description_embedding_store = None
            
            if self.graphrag_data_path:
                logger.warning(f"GraphRAG data path not found: {self.graphrag_data_path}, using context-only processing")
            else:
                logger.info("No GraphRAG data path provided, using context-only processing")
        
        logger.info(f"Initialized GraphRAG v2 system with model: {model_name}")
    
    def get_batch_size(self) -> int:
        """Return batch size for processing."""
        return 20  # Process multiple samples efficiently
    
    def _initialize_language_models(self):
        """Initialize language models for API calls."""
        if not self.use_api:
            logger.info("No API configuration provided, language models will be limited")
            return
        
        try:
            from graphrag.config.enums import ModelType
            from graphrag.config.models.language_model_config import LanguageModelConfig
            from graphrag.language_model.manager import ModelManager
            
            # Configure chat model for OpenAI-compatible API
            chat_config = LanguageModelConfig(
                api_key=self.api_key,
                type=ModelType.OpenAIChat,
                model=self.model_name,
                api_base=self.base_url,
                max_retries=20,
                encoding_model="cl100k_base",  # Explicitly specify encoding model
            )
            
            self.chat_model = ModelManager().get_or_create_chat_model(
                name="graphrag_local_search",
                model_type=ModelType.OpenAIChat,
                config=chat_config,
            )
            logger.info(f"Initialized chat model: {self.model_name}")
            
            # Configure embedding model
            embedding_config = LanguageModelConfig(
                api_key=self.api_key,
                type=ModelType.OpenAIEmbedding,
                model="text-embedding-ada-002",  # Standard embedding model name
                api_base=self.base_url,
                max_retries=20,
                encoding_model="cl100k_base",  # Explicitly specify encoding model
            )
            
            self.text_embedder = ModelManager().get_or_create_embedding_model(
                name="graphrag_embedding",
                model_type=ModelType.OpenAIEmbedding,
                config=embedding_config,
            )
            logger.info("Initialized text embedder")
            
        except Exception as e:
            logger.error(f"Error initializing language models: {e}")
            self.chat_model = None
            self.text_embedder = None
    
    def _extract_documents_from_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract documents from samples for indexing."""
        documents = []
        
        for sample in samples:
            sample_id = sample.get('id', f'sample_{len(documents)}')
            
            # Extract from search_results -> page_result
            if 'search_results' in sample and sample['search_results']:
                for i, result in enumerate(sample['search_results']):
                    if 'page_result' in result and result['page_result']:
                        documents.append({
                            'id': f"{sample_id}_doc_{i}",
                            'text': result['page_result'],
                            'sample_id': sample_id
                        })
            
            # Also include question as a document for context
            if 'question' in sample and sample['question']:
                documents.append({
                    'id': f"{sample_id}_question",
                    'text': sample['question'],
                    'sample_id': sample_id
                })
        
        logger.info(f"Extracted {len(documents)} documents from {len(samples)} samples")
        return documents
    
    def _prepare_documents_for_indexing(self, documents: List[Dict[str, Any]], output_dir: str):
        """Prepare documents as text files for GraphRAG indexing."""
        input_dir = Path(output_dir) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing files
        for file in input_dir.glob("*.txt"):
            file.unlink()
        
        # Save documents as text files
        for doc in documents:
            file_path = input_dir / f"{doc['id']}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc['text'])
        
        logger.info(f"Prepared {len(documents)} documents for indexing in {input_dir}")
    
    async def _index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents using GraphRAG."""
        try:
            # Create temporary directory for indexing
            output_dir = tempfile.mkdtemp(prefix="graphrag_v2_auto_")
            logger.info(f"Creating GraphRAG auto-index in: {output_dir}")
            
            # Prepare documents
            self._prepare_documents_for_indexing(documents, output_dir)
            
            # Create GraphRAG config for auto-indexing
            config_values = {
                "root_dir": output_dir,
                "input": {
                    "type": "files",
                    "glob": "**/*.txt",
                    "encoding": "utf-8"
                },
                "output": {
                    "base_dir": output_dir,
                    "format": "parquet"
                },
                # Explicitly set tokenizer encoding
                "encoding_model": "cl100k_base",
                "encoding": {"model": "cl100k_base"},
                "models": {
                    "default_chat_model": {
                        "type": "openai_chat",
                        "model": self.model_name,
                        "api_key": self.api_key,
                        "api_base": self.base_url,
                        "encoding_model": "cl100k_base",
                        "max_tokens": 2048,
                        "temperature": 0.1,
                        "model_supports_json": True
                    },
                    "default_embedding_model": {
                        "type": "openai_embedding",
                        "model": "text-embedding-ada-002",
                        "api_key": self.api_key,
                        "api_base": self.base_url,
                        "encoding_model": "cl100k_base"
                    }
                },
                "create_community_reports": {
                    "enabled": True  # Enable community reports for better search
                }
            }
            
            from graphrag.config.create_graphrag_config import create_graphrag_config
            from graphrag.config.enums import IndexingMethod
            from graphrag.api.index import build_index
            
            config = create_graphrag_config(values=config_values)
            
            # Run indexing
            logger.info("Starting GraphRAG auto-indexing...")
            results = await build_index(
                config=config,
                method=IndexingMethod.Standard,  # Use Standard method for full indexing
                is_update_run=False,
                memory_profile=False
            )
            
            # Check results
            success = all(not result.errors for result in results)
            if success:
                self.indexed_data_path = output_dir
                logger.info(f"GraphRAG auto-indexing completed successfully: {output_dir}")
                return True
            else:
                logger.error("GraphRAG auto-indexing failed with errors")
                for result in results:
                    if result.errors:
                        for error in result.errors:
                            logger.error(f"Auto-indexing error: {error}")
                return False
        
        except Exception as e:
            logger.error(f"Error during GraphRAG auto-indexing: {e}")
            return False
    
    def _load_indexed_data(self, data_path: str):
        """Load indexed data from GraphRAG output."""
        try:
            data_path = Path(data_path)
            
            # Load entities
            if (data_path / "entities.parquet").exists():
                entity_df = pd.read_parquet(data_path / "entities.parquet")
                community_df = pd.read_parquet(data_path / "communities.parquet") if (data_path / "communities.parquet").exists() else pd.DataFrame()
                self.entities = read_indexer_entities(entity_df, community_df, self.community_level)
                logger.info(f"Loaded {len(self.entities)} entities")
            else:
                self.entities = []
            
            # Load relationships
            if (data_path / "relationships.parquet").exists():
                relationship_df = pd.read_parquet(data_path / "relationships.parquet")
                self.relationships = read_indexer_relationships(relationship_df)
                logger.info(f"Loaded {len(self.relationships)} relationships")
            else:
                self.relationships = []
            
            # Load community reports
            if (data_path / "community_reports.parquet").exists():
                report_df = pd.read_parquet(data_path / "community_reports.parquet")
                entity_df = pd.read_parquet(data_path / "entities.parquet") if (data_path / "entities.parquet").exists() else pd.DataFrame()
                if not report_df.empty and not entity_df.empty:
                    self.reports = read_indexer_reports(report_df, entity_df, self.community_level)
                    logger.info(f"Loaded {len(self.reports)} community reports")
                else:
                    self.reports = []
            else:
                self.reports = []
            
            # Load text units
            if (data_path / "text_units.parquet").exists():
                text_unit_df = pd.read_parquet(data_path / "text_units.parquet")
                self.text_units = read_indexer_text_units(text_unit_df)
                logger.info(f"Loaded {len(self.text_units)} text units")
            else:
                self.text_units = []
            
            # Load covariates (claims) if available
            if (data_path / "covariates.parquet").exists():
                covariate_df = pd.read_parquet(data_path / "covariates.parquet")
                claims = read_indexer_covariates(covariate_df)
                self.covariates = {"claims": claims}
                logger.info(f"Loaded {len(claims)} claim records")
            else:
                self.covariates = None
            
            # Set up LanceDB vector store for entity embeddings
            lancedb_uri = data_path / "lancedb"
            if lancedb_uri.exists():
                self.description_embedding_store = LanceDBVectorStore(
                    collection_name="default-entity-description"
                )
                self.description_embedding_store.connect(db_uri=str(lancedb_uri))
                logger.info("Connected to auto-indexed LanceDB vector store")
            else:
                logger.warning("No LanceDB found in auto-indexed data")
                self.description_embedding_store = None
            
            self.indexed = True
            logger.info("Auto-indexed GraphRAG data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading auto-indexed data: {e}")
            # Set defaults to prevent crashes
            self.entities = []
            self.relationships = []
            self.reports = []
            self.text_units = []
            self.covariates = None
            self.indexed = False
    
    def _load_graphrag_data(self):
        """Load pre-indexed GraphRAG data following the provided example."""
        try:
            data_path = Path(self.graphrag_data_path)
            lancedb_uri = data_path / "lancedb"
            
            # Load entities
            entity_df = pd.read_parquet(data_path / "entities.parquet")
            community_df = pd.read_parquet(data_path / "communities.parquet")
            self.entities = read_indexer_entities(entity_df, community_df, self.community_level)
            logger.info(f"Loaded {len(self.entities)} entities")
            
            # Load relationships  
            relationship_df = pd.read_parquet(data_path / "relationships.parquet")
            self.relationships = read_indexer_relationships(relationship_df)
            logger.info(f"Loaded {len(self.relationships)} relationships")
            
            # Load community reports
            if (data_path / "community_reports.parquet").exists():
                report_df = pd.read_parquet(data_path / "community_reports.parquet")
                self.reports = read_indexer_reports(report_df, entity_df, self.community_level)
                logger.info(f"Loaded {len(self.reports)} community reports")
            else:
                self.reports = []
            
            # Load text units
            text_unit_df = pd.read_parquet(data_path / "text_units.parquet")
            self.text_units = read_indexer_text_units(text_unit_df)
            logger.info(f"Loaded {len(self.text_units)} text units")
            
            # Load covariates (claims) if available
            if (data_path / "covariates.parquet").exists():
                covariate_df = pd.read_parquet(data_path / "covariates.parquet")
                claims = read_indexer_covariates(covariate_df)
                self.covariates = {"claims": claims}
                logger.info(f"Loaded {len(claims)} claim records")
            else:
                self.covariates = None
            
            # Set up LanceDB vector store for entity embeddings
            self.description_embedding_store = LanceDBVectorStore(
                collection_name="default-entity-description"
            )
            if lancedb_uri.exists():
                self.description_embedding_store.connect(db_uri=str(lancedb_uri))
                logger.info("Connected to LanceDB vector store")
            else:
                logger.warning("LanceDB not found, will work without entity embeddings")
            
            logger.info("GraphRAG data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading GraphRAG data: {e}")
            # Set defaults to prevent crashes
            self.entities = []
            self.relationships = []
            self.reports = []
            self.text_units = []
            self.covariates = None
    
    def _create_context_specific_text_units(self, context_text: str, sample_id: str) -> List:
        """Create text units from the question's context text."""
        if not context_text:
            return []
        
        try:
            # Create a simple namespace object to match GraphRAG text unit structure
            class TextUnit:
                def __init__(self, id, text, n_tokens, document_ids=None, entity_ids=None, relationship_ids=None):
                    self.id = id
                    self.text = text
                    self.n_tokens = n_tokens
                    self.document_ids = document_ids or []
                    self.entity_ids = entity_ids or []
                    self.relationship_ids = relationship_ids or []
            
            text_unit = TextUnit(
                id=f"{sample_id}_context_unit",
                text=context_text,
                n_tokens=len(self.token_encoder.encode(context_text)),
                document_ids=[sample_id],
                entity_ids=[],
                relationship_ids=[]
            )
            
            return [text_unit]
        except Exception as e:
            logger.error(f"Error creating text units for {sample_id}: {e}")
            return []
    
    def _create_context_builder_for_sample(self, context_text: str, sample_id: str) -> Optional[LocalSearchMixedContext]:
        """Create a context builder for a specific question's context."""
        try:
            # Create context-specific text units
            context_text_units = self._create_context_specific_text_units(context_text, sample_id)
            
            # Combine with global knowledge graph data if available
            if hasattr(self, 'text_units') and self.text_units:
                combined_text_units = list(self.text_units) + context_text_units
            else:
                combined_text_units = context_text_units
            
            # Create context builder with question-specific text units
            # Only include entity_text_embeddings if we have a valid embedding store
            context_params = {
                'community_reports': getattr(self, 'reports', None) or [],
                'text_units': combined_text_units,
                'entities': getattr(self, 'entities', None) or [],
                'relationships': getattr(self, 'relationships', None) or [],
                'covariates': getattr(self, 'covariates', None),
                'text_embedder': getattr(self, 'text_embedder', None),
                'token_encoder': self.token_encoder,
            }
            
            # Only add embedding-related parameters if we have an embedding store
            embedding_store = getattr(self, 'description_embedding_store', None)
            if embedding_store is not None:
                context_params['entity_text_embeddings'] = embedding_store
                context_params['embedding_vectorstore_key'] = EntityVectorStoreKey.ID
            
            context_builder = LocalSearchMixedContext(**context_params)
            
            return context_builder
            
        except Exception as e:
            logger.error(f"Error creating context builder for {sample_id}: {e}")
            logger.exception(e)  # Add full traceback for debugging
            return None
    
    def _create_search_engine(self, context_builder: LocalSearchMixedContext) -> Optional[LocalSearch]:
        """Create a local search engine with the given context builder."""
        try:
            # Check if we have the required chat model
            if not self.chat_model:
                if self.use_api:
                    logger.warning("No chat model available for GraphRAG search (API mode), will skip search")
                else:
                    logger.warning("No chat model available for GraphRAG search (no API configured), will skip search")
                return None
            
            logger.info("Creating GraphRAG search engine with chat model")
            
            # Adjust context builder params based on available components
            context_params = self.local_context_params.copy()
            
            # If no embedding store, disable entity similarity search
            if not hasattr(self, 'description_embedding_store') or self.description_embedding_store is None:
                logger.info("No embedding store available, disabling entity similarity search")
                context_params["top_k_mapped_entities"] = 0  # Disable entity mapping
                # Remove embedding vectorstore key if not needed
                context_params.pop("embedding_vectorstore_key", None)
            
            search_engine = LocalSearch(
                context_builder=context_builder,
                token_encoder=self.token_encoder,
                model=self.chat_model,
                context_builder_params=context_params,
            )
            
            logger.info("GraphRAG search engine created successfully")
            return search_engine
            
        except Exception as e:
            logger.error(f"Error creating search engine: {e}")
            logger.exception(e)  # Add full traceback for debugging
            return None
    
    def _call_openai_api(self, prompt: str, max_tokens: int = 200, temperature: float = 0.1) -> Dict[str, Any]:
        """Call OpenAI-compatible API for text generation."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result['choices'][0]['message']['content'],
                    'usage': result.get('usage', {})
                }
            else:
                logger.error(f"API call failed with status {response.status_code}: {response.text}")
                return {'success': False, 'error': f"HTTP {response.status_code}: {response.text}"}
        
        except Exception as e:
            logger.error(f"API call error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_sample_with_api(self, sample: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Process sample using OpenAI-compatible API."""
        question = sample.get('question', '')
        options = sample.get('options', [])
        
        # Build prompt with context and question
        if context:
            prompt = f"""Context information:
{context}

Question: {question}

Please provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."""
        else:
            prompt = f"""Question: {question}

Please provide your final answer in the format Answer|X where X is your answer for the multiple choice question, which can be A, B, C, D, ..."""
        
        # Call API
        api_result = self._call_openai_api(prompt)
        
        if api_result['success']:
            response = api_result['response']
            
            # Extract answer from response
            import re
            answer_match = re.search(r'Answer\|([A-Z])', response, re.IGNORECASE)
            predicted_answer = answer_match.group(1).upper() if answer_match else "A"
            
            # Create simple probability distribution
            if options:
                # Assign high probability to predicted answer, low to others
                probabilities = {opt: 0.1 / (len(options) - 1) if opt != predicted_answer else 0.9 
                               for opt in options}
                if predicted_answer not in probabilities:
                    probabilities[predicted_answer] = 0.9
                    # Normalize
                    total = sum(probabilities.values())
                    probabilities = {k: v/total for k, v in probabilities.items()}
            else:
                probabilities = {predicted_answer: 1.0}
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': response,
                'predicted_answer': predicted_answer,
                'conformal_probabilities': probabilities,
                'technique': 'graphrag_v2_api',
                'api_usage': api_result.get('usage', {}),
                'system_type': 'graphrag_v2_api'
            }
        else:
            # API failed, return error
            return {
                'id': sample.get('id', 'unknown'),
                'error': f"API call failed: {api_result['error']}",
                'system_type': 'graphrag_v2_api_error'
            }
    
    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of samples with auto-indexing and GraphRAG local search."""
        
        # Auto-index if needed and enabled
        if not self.indexed and self.auto_index:
            logger.info("Auto-indexing samples for GraphRAG v2...")
            
            # Extract documents from samples
            documents = self._extract_documents_from_samples(samples)
            
            if documents:
                # Run async indexing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success = loop.run_until_complete(self._index_documents(documents))
                    if success and self.indexed_data_path:
                        self._load_indexed_data(self.indexed_data_path)
                finally:
                    loop.close()
            
            if not self.indexed:
                logger.warning("Failed to auto-index samples, falling back to context-only processing")
        
        # Check if we have GraphRAG data loaded (either pre-indexed or auto-indexed)
        if not (self.entities or self.text_units):
            logger.warning("No GraphRAG data available, using context-only processing")
        
        results = []
        
        for sample in samples:
            try:
                sample_id = sample.get('id', f'sample_{len(results)}')
                question = sample.get('question', '')
                
                # Extract context from search_results -> page_result
                context_text = ""
                if 'search_results' in sample and sample['search_results']:
                    context_parts = []
                    for result in sample['search_results']:
                        if 'page_result' in result and result['page_result']:
                            context_parts.append(result['page_result'])
                    context_text = "\n\n".join(context_parts)
                
                # Fallback to direct context field if available
                if not context_text and 'context' in sample:
                    context_text = sample['context']
                
                if not context_text:
                    logger.warning(f"No context text found for sample {sample_id}, using question only")
                    context_text = question
                
                # Create context builder for this specific question's context
                context_builder = self._create_context_builder_for_sample(context_text, sample_id)
                
                if context_builder:
                    # Create search engine for this context
                    search_engine = self._create_search_engine(context_builder)
                    
                    if search_engine:
                        # Perform local search with the question
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                search_result = loop.run_until_complete(search_engine.search(question))
                                graphrag_response = search_result.response if hasattr(search_result, 'response') else ""
                            finally:
                                loop.close()
                            
                            # Process through API if available
                            if self.use_api and graphrag_response:
                                result = self._process_sample_with_api(sample, graphrag_response)
                                result.update({
                                    'system_type': 'graphrag_v2_context_search',
                                    'graphrag_response': graphrag_response,
                                    'used_context_text': bool(context_text),
                                })
                            else:
                                # Create result with GraphRAG response as context
                                result = self._process_sample_with_api(sample, context_text) if self.use_api else {
                                    'id': sample_id,
                                    'generated_response': graphrag_response or context_text,
                                    'predicted_answer': self._extract_answer_from_text(graphrag_response or context_text),
                                    'conformal_probabilities': {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25},
                                    'technique': 'graphrag_v2_context',
                                    'system_type': 'graphrag_v2_context_only'
                                }
                                
                                result.update({
                                    'graphrag_response': graphrag_response,
                                    'used_context_text': bool(context_text),
                                })
                            
                        except Exception as search_error:
                            logger.error(f"GraphRAG search failed for {sample_id}: {search_error}")
                            # Fall back to simple context-based processing
                            result = self._process_sample_with_api(sample, context_text) if self.use_api else {
                                'id': sample_id,
                                'generated_response': context_text,
                                'predicted_answer': self._extract_answer_from_text(context_text),
                                'conformal_probabilities': {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25},
                                'system_type': 'graphrag_v2_context_fallback',
                                'error': str(search_error)
                            }
                    else:
                        # No search engine, use context directly
                        result = self._process_sample_with_api(sample, context_text) if self.use_api else {
                            'id': sample_id,
                            'generated_response': context_text,
                            'predicted_answer': self._extract_answer_from_text(context_text),
                            'conformal_probabilities': {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25},
                            'system_type': 'graphrag_v2_no_search_engine'
                        }
                else:
                    # No context builder, use simple processing
                    result = self._process_sample_with_api(sample, context_text) if self.use_api else {
                        'id': sample_id,
                        'generated_response': context_text,
                        'predicted_answer': self._extract_answer_from_text(context_text),
                        'conformal_probabilities': {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25},
                        'system_type': 'graphrag_v2_no_context_builder'
                    }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                # Create error result
                error_result = {
                    'id': sample.get('id', 'unknown'),
                    'error': str(e),
                    'system_type': 'graphrag_v2_error'
                }
                results.append(error_result)
        
        return results
    
    def _extract_answer_from_text(self, text: str) -> str:
        """Extract answer from text using simple pattern matching."""
        import re
        
        # Look for Answer|X pattern
        answer_match = re.search(r'Answer\|([A-Z])', text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Look for single letter answers
        letter_match = re.search(r'\b([A-Z])\b', text)
        if letter_match:
            return letter_match.group(1).upper()
        
        return "A"  # Default fallback
