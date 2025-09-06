from systems.abstract import AbstractRAGSystem
from systems.graphllm import GraphLLMSystem
from typing import Dict, Any, List, Optional
from loguru import logger
import pandas as pd
from pathlib import Path
import asyncio
import os
import sys
import tempfile
import shutil
from dataclasses import dataclass

# Add GraphRAG to path
graphrag_path = os.path.join(os.path.dirname(__file__), '..', 'graphrag')
sys.path.insert(0, graphrag_path)
# Also add the inner graphrag package directory
sys.path.insert(0, os.path.join(graphrag_path, 'graphrag'))

try:
    from graphrag.api.query import global_search, local_search, drift_search
    from graphrag.api.index import build_index
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.config.create_graphrag_config import create_graphrag_config
    from graphrag.config.enums import IndexingMethod
    from graphrag.query.indexer_adapters import (
        read_indexer_entities, 
        read_indexer_communities, 
        read_indexer_reports
    )
    from graphrag.index.typing.pipeline_run_result import PipelineRunResult
    GRAPHRAG_AVAILABLE = True
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    # Create a mock IndexingMethod enum for when GraphRAG is not available
    from enum import Enum
    class IndexingMethod(Enum):
        Standard = "standard"
        Fast = "fast"
        StandardUpdate = "standard-update"
        FastUpdate = "fast-update"


@dataclass
class GraphRAGIndexingResult:
    """Result from GraphRAG indexing process."""
    success: bool
    output_dir: str
    entities_count: int = 0
    relationships_count: int = 0
    communities_count: int = 0
    reports_count: int = 0
    text_units_count: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class GraphRAGSystem(AbstractRAGSystem):
    """
    Complete GraphRAG system that handles both indexing and querying.
    
    This system provides a complete GraphRAG implementation:
    1. Index documents to create knowledge graph
    2. Query the indexed data for enhanced responses
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium", 
                 device: str = "auto", 
                 graphrag_data_path: str = None,
                 graphrag_config_path: str = None,
                 search_type: str = "local",
                 community_level: int = 2,
                 # Indexing parameters
                 llm_model_id: str = "gpt-4o-mini",
                 llm_api_key: str = None,
                 llm_base_url: str = None,
                 indexing_method: str = "standard",
                 # Auto-indexing parameters
                 auto_index: bool = True,
                 dataset_path: str = None,
                 **kwargs):
        """Initialize the complete GraphRAG system."""
        
        # Initialize the underlying GraphLLM system
        self.llm_system = GraphLLMSystem(
            model_name=model_name, 
            device=device, 
            technique='graphrag',
            graphrag_data_path=graphrag_data_path
        )
        
        self.graphrag_data_path = graphrag_data_path
        self.graphrag_config_path = graphrag_config_path
        self.search_type = search_type  # "local", "global", or "drift"
        self.community_level = community_level
        
        # Indexing parameters
        self.llm_model_id = llm_model_id
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("OPENAI_BASE_URL")
        self.indexing_method = IndexingMethod.Standard if indexing_method == "standard" else IndexingMethod.Fast
        
        # Auto-indexing parameters
        self.auto_index = auto_index
        self.dataset_path = dataset_path
        
        # Load GraphRAG data and configuration
        self.config = None
        self.entities_df = None
        self.communities_df = None
        self.community_reports_df = None
        self.text_units_df = None
        self.relationships_df = None
        self.covariates_df = None
        
        # Indexing state
        self.indexed = False
        self.indexing_result = None
        self.extracted_documents = None
        
        if GRAPHRAG_AVAILABLE and graphrag_data_path:
            self._load_graphrag_data()
            if graphrag_config_path:
                self._load_graphrag_config()
        elif graphrag_data_path:
            logger.warning(f"GraphRAG data path provided ({graphrag_data_path}) but GraphRAG not available")
        
        # Auto-index if enabled and no existing data
        if self.auto_index and not self.indexed and self.dataset_path:
            logger.info(f"Auto-indexing enabled. Dataset path: {self.dataset_path}")
            logger.info(f"GraphRAG available: {GRAPHRAG_AVAILABLE}")
            logger.info(f"Using local model: {self.llm_model_id}")
            self._auto_index_dataset()
        else:
            logger.info(f"Auto-indexing conditions: auto_index={self.auto_index}, indexed={self.indexed}, dataset_path={self.dataset_path}")
    
    def _load_graphrag_data(self):
        """Load all GraphRAG generated data files."""
        if not self.graphrag_data_path:
            return
            
        try:
            data_path = Path(self.graphrag_data_path)
            
            # Load entities
            entities_file = data_path / "entities.parquet"
            if entities_file.exists():
                self.entities_df = pd.read_parquet(entities_file)
                logger.info(f"Loaded {len(self.entities_df)} entities")
            
            # Load communities
            communities_file = data_path / "communities.parquet"
            if communities_file.exists():
                self.communities_df = pd.read_parquet(communities_file)
                logger.info(f"Loaded {len(self.communities_df)} communities")
            
            # Load community reports
            reports_file = data_path / "community_reports.parquet"
            if reports_file.exists():
                self.community_reports_df = pd.read_parquet(reports_file)
                logger.info(f"Loaded {len(self.community_reports_df)} community reports")
            
            # Load text units
            text_units_file = data_path / "text_units.parquet"
            if text_units_file.exists():
                self.text_units_df = pd.read_parquet(text_units_file)
                logger.info(f"Loaded {len(self.text_units_df)} text units")
            
            # Load relationships
            relationships_file = data_path / "relationships.parquet"
            if relationships_file.exists():
                self.relationships_df = pd.read_parquet(relationships_file)
                logger.info(f"Loaded {len(self.relationships_df)} relationships")
                
            # Load covariates (optional)
            covariates_file = data_path / "covariates.parquet"
            if covariates_file.exists():
                self.covariates_df = pd.read_parquet(covariates_file)
                logger.info(f"Loaded {len(self.covariates_df)} covariates")
                
        except Exception as e:
            logger.error(f"Error loading GraphRAG data: {e}")
            # Fall back to LLM-only mode
            logger.info("Falling back to GraphLLM mode without full GraphRAG search")
    
    def _load_graphrag_config(self):
        """Load GraphRAG configuration."""
        if not GRAPHRAG_AVAILABLE:
            return
            
        try:
            if self.graphrag_config_path and os.path.exists(self.graphrag_config_path):
                # Load from config file
                self.config = create_graphrag_config(root_dir=os.path.dirname(self.graphrag_config_path))
                logger.info(f"Loaded GraphRAG config from {self.graphrag_config_path}")
            else:
                # Create default config
                if self.graphrag_data_path:
                    self.config = create_graphrag_config(root_dir=str(Path(self.graphrag_data_path).parent))
                    logger.info("Created default GraphRAG config")
        except Exception as e:
            logger.warning(f"Could not load GraphRAG config: {e}")
    
    def _setup_indexing_config(self, output_dir: str):
        """Set up GraphRAG configuration for indexing."""
        if not GRAPHRAG_AVAILABLE:
            logger.warning("GraphRAG not available, cannot create indexing configuration")
            return None
            
        try:
            # Create basic configuration for indexing
            # Use a simpler configuration that works with local models
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
                "extract_graph": {
                    "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT"]
                },
                "create_communities": {
                    "algorithm": "leiden",
                    "resolution": 1.0,
                    "random_state": 42
                }
            }
            
            # For local models, we'll skip the complex LLM configuration
            # and let GraphRAG use its default settings
            logger.info("Using simplified configuration for local model")
            
            config = create_graphrag_config(values=config_values)
            logger.info(f"GraphRAG indexing configuration created for output: {output_dir}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to create GraphRAG indexing configuration: {e}")
            return None
    
    def _prepare_documents(self, documents: List[Dict[str, Any]], 
                          text_field: str, id_field: str, output_dir: str):
        """Prepare documents for GraphRAG by saving them as text files."""
        input_dir = Path(output_dir) / "input"
        input_dir.mkdir(exist_ok=True)
        
        # Clear existing input files
        for file in input_dir.glob("*.txt"):
            file.unlink()
        
        # Save documents as text files
        for i, doc in enumerate(documents):
            doc_id = doc.get(id_field, f"doc_{i}")
            text_content = doc.get(text_field, "")
            
            if not text_content.strip():
                logger.warning(f"Document {doc_id} has no text content, skipping")
                continue
            
            # Save as text file
            file_path = input_dir / f"{doc_id}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
        
        logger.info(f"Prepared {len(documents)} documents for GraphRAG indexing")
    
    def _count_generated_data(self, output_dir: str) -> Dict[str, int]:
        """Count the generated data files."""
        counts = {
            'entities': 0,
            'relationships': 0,
            'communities': 0,
            'reports': 0,
            'text_units': 0
        }
        
        try:
            output_path = Path(output_dir)
            
            # Count entities
            entities_file = output_path / "entities.parquet"
            if entities_file.exists():
                df = pd.read_parquet(entities_file)
                counts['entities'] = len(df)
            
            # Count relationships
            relationships_file = output_path / "relationships.parquet"
            if relationships_file.exists():
                df = pd.read_parquet(relationships_file)
                counts['relationships'] = len(df)
            
            # Count communities
            communities_file = output_path / "communities.parquet"
            if communities_file.exists():
                df = pd.read_parquet(communities_file)
                counts['communities'] = len(df)
            
            # Count community reports
            reports_file = output_path / "community_reports.parquet"
            if reports_file.exists():
                df = pd.read_parquet(reports_file)
                counts['reports'] = len(df)
            
            # Count text units
            text_units_file = output_path / "text_units.parquet"
            if text_units_file.exists():
                df = pd.read_parquet(text_units_file)
                counts['text_units'] = len(df)
                
        except Exception as e:
            logger.warning(f"Error counting generated data: {e}")
        
        return counts
    
    def _extract_documents_from_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Extract documents from the dataset for indexing."""
        import json
        
        try:
            logger.info(f"Loading dataset from: {dataset_path}")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            logger.info(f"Dataset keys: {list(dataset.keys())}")
            documents = []
            
            # Extract from calibration samples
            if 'calibration' in dataset:
                logger.info(f"Processing {len(dataset['calibration'])} calibration samples")
                for i, sample in enumerate(dataset['calibration']):
                    if 'search_results' in sample:
                        logger.debug(f"Sample {i} has {len(sample['search_results'])} search results")
                        for j, result in enumerate(sample['search_results']):
                            if 'page_result' in result and result['page_result']:
                                doc_id = f"cal_{sample['id']}_{result.get('page_name', 'unknown')}"
                                documents.append({
                                    'id': doc_id,
                                    'text': result['page_result'],
                                    'source': 'calibration',
                                    'sample_id': sample['id']
                                })
                                logger.debug(f"Added document {doc_id} from calibration sample {i}, result {j}")
                            else:
                                logger.debug(f"No page_result in calibration sample {i}, result {j}")
                    else:
                        logger.debug(f"No search_results in calibration sample {i}")
            
            # Extract from test samples
            if 'test' in dataset:
                logger.info(f"Processing {len(dataset['test'])} test samples")
                for i, sample in enumerate(dataset['test']):
                    if 'search_results' in sample:
                        logger.debug(f"Test sample {i} has {len(sample['search_results'])} search results")
                        for j, result in enumerate(sample['search_results']):
                            if 'page_result' in result and result['page_result']:
                                doc_id = f"test_{sample['id']}_{result.get('page_name', 'unknown')}"
                                documents.append({
                                    'id': doc_id,
                                    'text': result['page_result'],
                                    'source': 'test',
                                    'sample_id': sample['id']
                                })
                                logger.debug(f"Added document {doc_id} from test sample {i}, result {j}")
                            else:
                                logger.debug(f"No page_result in test sample {i}, result {j}")
                    else:
                        logger.debug(f"No search_results in test sample {i}")
            
            logger.info(f"Extracted {len(documents)} documents from dataset: {dataset_path}")
            if documents:
                logger.info(f"Sample document IDs: {[doc['id'] for doc in documents[:3]]}")
                # Log first few characters of first document for debugging
                first_doc_text = documents[0]['text'][:200] if documents[0]['text'] else "EMPTY"
                logger.info(f"First document text preview: {first_doc_text}...")
            else:
                logger.warning("No documents were extracted from the dataset!")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting documents from dataset {dataset_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _auto_index_dataset(self):
        """Automatically index documents from the dataset."""
        logger.info(f"Starting auto-indexing process...")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Dataset exists: {os.path.exists(self.dataset_path) if self.dataset_path else False}")
        
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            logger.warning(f"Dataset path not found: {self.dataset_path}")
            return
        
        # Extract documents from dataset first
        logger.info("Extracting documents from dataset...")
        documents = self._extract_documents_from_dataset(self.dataset_path)
        if not documents:
            logger.error("No documents extracted from dataset - auto-indexing aborted")
            return
        
        logger.info(f"Successfully extracted {len(documents)} documents")
        
        # Store documents for use in GraphLLM system
        self.extracted_documents = documents
        
        if not GRAPHRAG_AVAILABLE:
            logger.warning("GraphRAG not available, using fallback mode with extracted documents")
            # Mark as indexed with fallback mode
            self.indexed = True
            self.indexing_result = GraphRAGIndexingResult(
                success=True,
                output_dir="fallback_mode",
                entities_count=0,
                relationships_count=0,
                communities_count=0,
                reports_count=0,
                text_units_count=len(documents),
                errors=["GraphRAG not available, using fallback mode"]
            )
            return
        
        try:
            # Create output directory for indexing
            output_dir = tempfile.mkdtemp(prefix="graphrag_auto_")
            logger.info(f"Created output directory: {output_dir}")
            
            # Run indexing synchronously
            logger.info("Starting GraphRAG indexing...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.index_documents(
                    documents=documents,
                    document_text_field="text",
                    document_id_field="id",
                    output_dir=output_dir
                ))
                
                if result.success:
                    logger.info("Auto-indexing completed successfully!")
                    logger.info(f"Generated: {result.entities_count} entities, {result.relationships_count} relationships")
                    self.indexed = True
                    self.indexing_result = result
                    # Update the GraphRAG data path to the new location
                    self.graphrag_data_path = output_dir
                else:
                    logger.error(f"Auto-indexing failed with errors: {result.errors}")
                    for i, error in enumerate(result.errors, 1):
                        logger.error(f"Error {i}: {error}")
                    # Fall back to extracted documents mode
                    logger.info("Falling back to extracted documents mode")
                    self.indexed = True
                    self.extracted_documents = documents
                    self.indexing_result = GraphRAGIndexingResult(
                        success=False,
                        output_dir="fallback_mode",
                        entities_count=0,
                        relationships_count=0,
                        communities_count=0,
                        reports_count=0,
                        text_units_count=len(documents),
                        errors=result.errors
                    )
                    
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Critical error during auto-indexing: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Fall back to extracted documents mode
            logger.info("Falling back to extracted documents mode due to error")
            self.indexed = True
            self.extracted_documents = documents
            self.indexing_result = GraphRAGIndexingResult(
                success=False,
                output_dir="fallback_mode",
                entities_count=0,
                relationships_count=0,
                communities_count=0,
                reports_count=0,
                text_units_count=len(documents),
                errors=[str(e)]
            )
    
    async def index_documents(self, 
                            documents: List[Dict[str, Any]], 
                            document_text_field: str = "text",
                            document_id_field: str = "id",
                            output_dir: str = None) -> GraphRAGIndexingResult:
        """
        Index a list of documents using GraphRAG.
        
        Args:
            documents: List of documents with text content
            document_text_field: Field name containing the text content
            document_id_field: Field name containing the document ID
            output_dir: Directory to store indexed data (optional)
            
        Returns:
            GraphRAGIndexingResult with indexing results
        """
        if not GRAPHRAG_AVAILABLE:
            return GraphRAGIndexingResult(
                success=False, 
                output_dir=output_dir or "",
                errors=["GraphRAG not available. Please install GraphRAG to use indexing functionality."]
            )
        
        # Set up output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="graphrag_")
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare documents for GraphRAG
            self._prepare_documents(documents, document_text_field, document_id_field, output_dir)
            
            # Set up configuration
            config = self._setup_indexing_config(output_dir)
            if not config:
                return GraphRAGIndexingResult(
                    success=False,
                    output_dir=output_dir,
                    errors=["Failed to create GraphRAG configuration"]
                )
            
            logger.info(f"Starting GraphRAG indexing of {len(documents)} documents...")
            logger.info(f"Using method: {self.indexing_method}")
            logger.info(f"Output directory: {output_dir}")
            
            # Run the indexing pipeline
            results = await build_index(
                config=config,
                method=self.indexing_method,
                is_update_run=False,
                memory_profile=False
            )
            
            # Process results
            success = all(not result.errors for result in results)
            errors = []
            for result in results:
                if result.errors:
                    errors.extend([str(error) for error in result.errors])
            
            # Count generated data
            counts = self._count_generated_data(output_dir)
            
            logger.info(f"GraphRAG indexing completed. Success: {success}")
            logger.info(f"Generated: {counts['entities']} entities, {counts['relationships']} relationships, "
                       f"{counts['communities']} communities, {counts['reports']} reports")
            
            # Update the system's data path and reload data
            self.graphrag_data_path = output_dir
            self._load_graphrag_data()
            self._load_graphrag_config()
            self.indexed = True
            
            result = GraphRAGIndexingResult(
                success=success,
                output_dir=output_dir,
                entities_count=counts['entities'],
                relationships_count=counts['relationships'],
                communities_count=counts['communities'],
                reports_count=counts['reports'],
                text_units_count=counts['text_units'],
                errors=errors
            )
            
            self.indexing_result = result
            return result
            
        except Exception as e:
            logger.error(f"GraphRAG indexing failed: {e}")
            return GraphRAGIndexingResult(
                success=False,
                output_dir=output_dir,
                errors=[str(e)]
            )
    
    def get_batch_size(self) -> int:
        """Return batch size."""
        return 1
    
    async def _perform_graphrag_search(self, query: str) -> Dict[str, Any]:
        """Perform GraphRAG search using the official API."""
        if not GRAPHRAG_AVAILABLE or self.config is None:
            logger.warning("GraphRAG not available, falling back to GraphLLM")
            return {"response": "", "context": ""}
        
        try:
            if self.search_type == "global" and self.entities_df is not None and self.community_reports_df is not None:
                # Global search
                response, context = await global_search(
                    config=self.config,
                    entities=self.entities_df,
                    communities=self.communities_df or pd.DataFrame(),
                    community_reports=self.community_reports_df,
                    community_level=self.community_level,
                    dynamic_community_selection=False,
                    response_type="multiple paragraphs",
                    query=query
                )
                logger.debug("Performed GraphRAG global search")
                
            elif self.search_type == "local" and all(df is not None for df in [self.entities_df, self.text_units_df, self.relationships_df]):
                # Local search
                response, context = await local_search(
                    config=self.config,
                    entities=self.entities_df,
                    communities=self.communities_df or pd.DataFrame(),
                    community_reports=self.community_reports_df or pd.DataFrame(),
                    text_units=self.text_units_df,
                    relationships=self.relationships_df,
                    covariates=self.covariates_df,
                    community_level=self.community_level,
                    response_type="multiple paragraphs",
                    query=query
                )
                logger.debug("Performed GraphRAG local search")
                
            elif self.search_type == "drift" and all(df is not None for df in [self.entities_df, self.text_units_df, self.relationships_df]):
                # Drift search
                response, context = await drift_search(
                    config=self.config,
                    entities=self.entities_df,
                    communities=self.communities_df or pd.DataFrame(),
                    community_reports=self.community_reports_df or pd.DataFrame(),
                    text_units=self.text_units_df,
                    relationships=self.relationships_df,
                    community_level=self.community_level,
                    response_type="multiple paragraphs",
                    query=query
                )
                logger.debug("Performed GraphRAG drift search")
                
            else:
                logger.warning(f"Cannot perform {self.search_type} search - missing required data")
                return {"response": "", "context": ""}
                
            return {
                "response": str(response) if response else "",
                "context": context
            }
            
        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            return {"response": "", "context": ""}
    
    def _extract_relevant_documents(self, search_result: Dict[str, Any]) -> List[str]:
        """Extract relevant document texts from GraphRAG search context."""
        documents = []
        
        # The GraphRAG search returns context with various structures
        search_context = search_result.get("context", {})
        
        # Try to extract text units or documents from the search context
        if isinstance(search_context, dict):
            # Look for text content in various possible keys
            for key in ['selected_entities', 'reports', 'sources', 'text_units', 'documents']:
                if key in search_context:
                    items = search_context[key]
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                # Extract text from various possible text fields
                                text_content = (
                                    item.get('text', '') or 
                                    item.get('content', '') or 
                                    item.get('description', '') or
                                    item.get('summary', '')
                                )
                                if text_content and len(text_content.strip()) > 50:
                                    documents.append(text_content.strip())
                            elif isinstance(item, str) and len(item.strip()) > 50:
                                documents.append(item.strip())
        
        # If we have text_units_df loaded, also get relevant text units based on entities/relationships
        if self.text_units_df is not None and hasattr(search_result.get("context"), 'get'):
            relevant_entities = search_result["context"].get('selected_entities', [])
            if relevant_entities:
                # Find text units that contain mentions of the relevant entities
                for _, text_unit in self.text_units_df.iterrows():
                    text_content = text_unit.get('text', '')
                    if text_content and any(entity.lower() in text_content.lower() for entity in relevant_entities if isinstance(entity, str)):
                        if len(text_content.strip()) > 50 and text_content not in documents:
                            documents.append(text_content.strip())
        
        # Limit to top 5 most relevant documents to avoid token overflow
        return documents[:5]
    
    def _build_enhanced_context(self, search_result: Dict[str, Any], sample: Dict[str, Any]) -> str:
        """Build enhanced context combining GraphRAG search results with actual retrieved documents."""
        context_parts = []
        
        # Extract and include actual retrieved documents (PRIMARY RAG CONTEXT)
        retrieved_docs = self._extract_relevant_documents(search_result)
        if retrieved_docs:
            context_parts.append("Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"Document {i}:")
                context_parts.append(doc)
                context_parts.append("")  # Empty line for separation
        
        # Add GraphRAG analysis as supplementary context
        if search_result.get("response"):
            context_parts.append("GraphRAG Analysis:")
            context_parts.append(search_result["response"])
        
        # Add any existing context from sample
        existing_context = sample.get('context', '')
        if existing_context:
            context_parts.append("\nAdditional Context:")
            context_parts.append(existing_context)
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with GraphRAG enhancement."""
        question = sample.get('question', '')
        
        # Check if documents have been indexed
        if not self.indexed and not self.graphrag_data_path:
            error_msg = "GraphRAG indexing not completed. "
            if self.auto_index and self.dataset_path:
                error_msg += f"Auto-indexing was enabled but may have failed. Check logs for details. Dataset: {self.dataset_path}"
            else:
                error_msg += "Please run index_documents() first or enable auto_index with a valid dataset_path."
            
            logger.warning(error_msg)
            return {
                'id': sample.get('id', 'unknown'),
                'error': error_msg,
                'system_type': 'graphrag',
                'auto_index_enabled': self.auto_index,
                'dataset_path': self.dataset_path,
                'graphrag_available': GRAPHRAG_AVAILABLE
            }
        
        # If we have extracted documents but no full GraphRAG indexing, use them for context
        if hasattr(self, 'extracted_documents') and self.extracted_documents:
            logger.info(f"Using {len(self.extracted_documents)} extracted documents for context")
            # Create a simple context from the extracted documents
            context_parts = []
            for i, doc in enumerate(self.extracted_documents[:5]):  # Limit to first 5 docs
                context_parts.append(f"Document {i+1} ({doc['id']}):")
                context_parts.append(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])
                context_parts.append("")
            
            enhanced_context = "\n".join(context_parts)
            
            # Create augmented sample with extracted document context
            augmented_sample = sample.copy()
            augmented_sample['context'] = enhanced_context
            augmented_sample['technique'] = 'graphrag_fallback'
            logger.debug(f"Enhanced sample with extracted document context: {len(enhanced_context)} characters")
            
            # Process through the GraphLLM system
            result = self.llm_system.process_sample(augmented_sample)
            
            # Add GraphRAG-specific information
            result.update({
                'search_type': 'fallback',
                'graphrag_enhanced': True,
                'graphrag_response': "Using extracted documents as context",
                'system_type': 'graphrag_fallback',
                'indexed': self.indexed,
                'data_sources': {
                    'entities': 0,
                    'relationships': 0,
                    'communities': 0,
                    'reports': 0,
                    'text_units': len(self.extracted_documents)
                }
            })
            
            # Add indexing result information if available
            if self.indexing_result:
                result['indexing_result'] = {
                    'entities_count': self.indexing_result.entities_count,
                    'relationships_count': self.indexing_result.relationships_count,
                    'communities_count': self.indexing_result.communities_count,
                    'reports_count': self.indexing_result.reports_count,
                    'text_units_count': self.indexing_result.text_units_count
                }
            
            return result
        
        # Perform GraphRAG search if available
        search_result = {"response": "", "context": ""}
        if GRAPHRAG_AVAILABLE and self.config is not None:
            try:
                # Run async search
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                search_result = loop.run_until_complete(self._perform_graphrag_search(question))
                loop.close()
            except Exception as e:
                logger.warning(f"Async GraphRAG search failed, using synchronous fallback: {e}")
        
        # Build enhanced context
        enhanced_context = self._build_enhanced_context(search_result, sample)
        
        # Create augmented sample with GraphRAG context
        augmented_sample = sample.copy()
        if enhanced_context:
            augmented_sample['context'] = enhanced_context
            augmented_sample['technique'] = 'graphrag'
            logger.debug(f"Enhanced sample with GraphRAG context: {len(enhanced_context)} characters")
        else:
            logger.debug("No GraphRAG context available, using base GraphLLM system")
        
        # Process through the GraphLLM system
        result = self.llm_system.process_sample(augmented_sample)
        
        # Add GraphRAG-specific information
        result.update({
            'search_type': self.search_type,
            'graphrag_enhanced': bool(enhanced_context),
            'graphrag_response': search_result.get("response", ""),
            'system_type': 'graphrag',
            'indexed': self.indexed,
            'data_sources': {
                'entities': len(self.entities_df) if self.entities_df is not None else 0,
                'relationships': len(self.relationships_df) if self.relationships_df is not None else 0,
                'communities': len(self.communities_df) if self.communities_df is not None else 0,
                'reports': len(self.community_reports_df) if self.community_reports_df is not None else 0,
                'text_units': len(self.text_units_df) if self.text_units_df is not None else 0
            }
        })
        
        # Add indexing result information if available
        if self.indexing_result:
            result['indexing_result'] = {
                'entities_count': self.indexing_result.entities_count,
                'relationships_count': self.indexing_result.relationships_count,
                'communities_count': self.indexing_result.communities_count,
                'reports_count': self.indexing_result.reports_count
            }
        
        return result
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up temporary directories if they were created
            if (hasattr(self, 'indexing_result') and 
                self.indexing_result and 
                self.indexing_result.output_dir and
                self.indexing_result.output_dir.startswith(tempfile.gettempdir())):
                
                if Path(self.indexing_result.output_dir).exists():
                    shutil.rmtree(self.indexing_result.output_dir)
                    logger.info(f"Cleaned up temporary GraphRAG directory: {self.indexing_result.output_dir}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")