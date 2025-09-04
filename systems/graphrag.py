from systems.abstract import AbstractRAGSystem
from systems.graphllm import GraphLLMSystem
from typing import Dict, Any, List
from loguru import logger
import pandas as pd
from pathlib import Path
import asyncio
import os
import sys

# Add GraphRAG to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graphrag'))

try:
    from graphrag.api.query import global_search, local_search, drift_search
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.config.create_graphrag_config import create_graphrag_config
    from graphrag.query.indexer_adapters import (
        read_indexer_entities, 
        read_indexer_communities, 
        read_indexer_reports
    )
    GRAPHRAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GraphRAG library not fully available: {e}")
    GRAPHRAG_AVAILABLE = False


class GraphRAGSystem(AbstractRAGSystem):
    """
    Full GraphRAG system that performs knowledge graph-based retrieval and augmentation.
    
    This demonstrates how to implement a GraphRAG system with global/local search capabilities,
    following the existing patterns from SimpleRAG and SimpleLLM.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium", 
                 device: str = "auto", 
                 graphrag_data_path: str = None,
                 graphrag_config_path: str = None,
                 search_type: str = "local",
                 community_level: int = 2,
                 **kwargs):
        """Initialize the GraphRAG system."""
        
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
        
        # Load GraphRAG data and configuration
        self.config = None
        self.entities_df = None
        self.communities_df = None
        self.community_reports_df = None
        self.text_units_df = None
        self.relationships_df = None
        self.covariates_df = None
        
        if GRAPHRAG_AVAILABLE and graphrag_data_path:
            self._load_graphrag_data()
            if graphrag_config_path:
                self._load_graphrag_config()
    
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
    
    def _build_enhanced_context(self, search_result: Dict[str, Any], sample: Dict[str, Any]) -> str:
        """Build enhanced context combining GraphRAG search results with sample data."""
        context_parts = []
        
        # Add GraphRAG search response as context
        if search_result.get("response"):
            context_parts.append("GraphRAG Analysis:")
            context_parts.append(search_result["response"])
        
        # Add any existing context from sample
        existing_context = sample.get('context', '')
        if existing_context:
            context_parts.append("\nAdditional Context:")
            context_parts.append(existing_context)
        
        # Add search metadata if available
        if isinstance(search_result.get("context"), dict):
            context_parts.append("\nSearch Metadata:")
            for key, value in search_result["context"].items():
                if isinstance(value, (list, pd.DataFrame)) and hasattr(value, '__len__'):
                    context_parts.append(f"- {key}: {len(value)} items")
                else:
                    context_parts.append(f"- {key}: {str(value)[:100]}...")
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with GraphRAG enhancement."""
        question = sample.get('question', '')
        
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
            'data_sources': {
                'entities': len(self.entities_df) if self.entities_df is not None else 0,
                'relationships': len(self.relationships_df) if self.relationships_df is not None else 0,
                'communities': len(self.communities_df) if self.communities_df is not None else 0,
                'reports': len(self.community_reports_df) if self.community_reports_df is not None else 0,
                'text_units': len(self.text_units_df) if self.text_units_df is not None else 0
            }
        })
        
        return result