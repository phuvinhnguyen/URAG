from systems.abstract import AbstractRAGSystem
from systems.simplellm import SimpleLLMSystem
from typing import Dict, Any, List, Optional
from loguru import logger
import pandas as pd
import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Add GraphRAG to path for compatibility
graphrag_path = os.path.join(os.path.dirname(__file__), '..', 'graphrag')
sys.path.insert(0, graphrag_path)
sys.path.insert(0, os.path.join(graphrag_path, 'graphrag'))

try:
    from graphrag.query.indexer_adapters import (
        read_indexer_entities,
        read_indexer_relationships,
        read_indexer_reports,
        read_indexer_text_units,
    )
    from graphrag.api.index import build_index
    from graphrag.config.create_graphrag_config import create_graphrag_config
    from graphrag.config.enums import IndexingMethod
    GRAPHRAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GraphRAG not available: {e}")
    GRAPHRAG_AVAILABLE = False


class GraphLLMSystem(AbstractRAGSystem):
    """
    GraphLLM v2 system - Direct LLM with GraphRAG knowledge enhancement.
    
    Unlike GraphRAG v2 which uses retrieval, this system directly incorporates
    knowledge graph information into prompts without retrieval-based RAG.
    It follows the SimpleLLM pattern but with graph-based context enhancement.
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
                 device: str = "cuda",
                 # Graph parameters
                 community_level: int = 2,
                 max_entities: int = 10,
                 max_relationships: int = 10,
                 max_reports: int = 3,
                 # Indexing parameters
                 auto_index: bool = True,
                 **kwargs):
        """Initialize GraphLLM v2 system."""
        
        # Initialize underlying LLM system
        self.llm_system = SimpleLLMSystem(
            model_name=model_name,
            device=device,
            technique='direct',  # Direct prompting, not RAG
            **kwargs
        )
        
        # Graph configuration
        self.community_level = community_level
        self.max_entities = max_entities
        self.max_relationships = max_relationships
        self.max_reports = max_reports
        
        # Storage for indexed data
        self.entities = None
        self.relationships = None
        self.reports = None
        self.text_units = None
        self.entities_df = None
        self.relationships_df = None
        self.reports_df = None
        
        # Indexing state
        self.indexed = False
        self.indexed_data_path = None
        self.auto_index = auto_index
    
    def get_batch_size(self) -> int:
        """Return batch size for processing."""
        return 20  # Process multiple samples efficiently
    
    def _extract_documents_from_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract documents from samples for indexing."""
        documents = []
        
        for sample in samples:
            sample_id = sample.get('id', f'sample_{len(documents)}')
            
            # Extract from search_results if available
            if 'search_results' in sample:
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
        if not GRAPHRAG_AVAILABLE:
            logger.error("GraphRAG not available for indexing")
            return False
        
        try:
            # Create temporary directory for indexing
            output_dir = tempfile.mkdtemp(prefix="graphllm_v2_")
            logger.info(f"Creating GraphRAG index in: {output_dir}")
            
            # Prepare documents
            self._prepare_documents_for_indexing(documents, output_dir)
            
            # Create GraphRAG config
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
                }
            }
            
            config = create_graphrag_config(values=config_values)
            
            # Run indexing
            logger.info("Starting GraphRAG indexing...")
            results = await build_index(
                config=config,
                method=IndexingMethod.Standard,
                is_update_run=False,
                memory_profile=False
            )
            
            # Check results
            success = all(not result.errors for result in results)
            if success:
                self.indexed_data_path = output_dir
                logger.info(f"GraphRAG indexing completed successfully: {output_dir}")
                return True
            else:
                logger.error("GraphRAG indexing failed with errors")
                for result in results:
                    if result.errors:
                        for error in result.errors:
                            logger.error(f"Indexing error: {error}")
                return False
        
        except Exception as e:
            logger.error(f"Error during GraphRAG indexing: {e}")
            return False
    
    def _load_indexed_data(self, data_path: str):
        """Load indexed data from GraphRAG output."""
        try:
            data_path = Path(data_path)
            
            # Load raw dataframes
            self.entities_df = pd.read_parquet(data_path / "entities.parquet")
            self.relationships_df = pd.read_parquet(data_path / "relationships.parquet")
            
            # Load community reports if available
            if (data_path / "community_reports.parquet").exists():
                self.reports_df = pd.read_parquet(data_path / "community_reports.parquet")
                logger.info(f"Loaded {len(self.reports_df)} community reports")
            else:
                self.reports_df = pd.DataFrame()
            
            # Also load processed objects for compatibility
            community_df = pd.read_parquet(data_path / "communities.parquet") if (data_path / "communities.parquet").exists() else pd.DataFrame()
            self.entities = read_indexer_entities(self.entities_df, community_df, self.community_level)
            self.relationships = read_indexer_relationships(self.relationships_df)
            
            if not self.reports_df.empty:
                self.reports = read_indexer_reports(self.reports_df, self.entities_df, self.community_level)
            else:
                self.reports = []
            
            self.indexed = True
            logger.info(f"Loaded GraphRAG data: {len(self.entities)} entities, {len(self.relationships)} relationships, {len(self.reports)} reports")
            
        except Exception as e:
            logger.error(f"Error loading indexed data: {e}")
            self.indexed = False
    
    def _find_relevant_entities(self, question: str) -> List[Dict[str, Any]]:
        """Find entities relevant to the question using simple keyword matching."""
        if self.entities_df is None or self.entities_df.empty:
            return []
        
        question_lower = question.lower()
        relevant_entities = []
        
        for _, entity in self.entities_df.iterrows():
            entity_title = str(entity.get('title', '')).lower()
            entity_desc = str(entity.get('description', '')).lower()
            
            # Simple relevance scoring
            relevance_score = 0
            
            # Title matches get high score
            if entity_title in question_lower or any(word in entity_title for word in question_lower.split() if len(word) > 2):
                relevance_score += 3
            
            # Description matches get lower score
            if any(word in entity_desc for word in question_lower.split() if len(word) > 2):
                relevance_score += 1
            
            if relevance_score > 0:
                relevant_entities.append({
                    'title': entity.get('title', ''),
                    'description': entity.get('description', ''),
                    'type': entity.get('type', 'ENTITY'),
                    'relevance_score': relevance_score
                })
        
        # Sort by relevance and return top entities
        relevant_entities.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_entities[:self.max_entities]
    
    def _find_relevant_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find relationships involving the relevant entities."""
        if self.relationships_df is None or self.relationships_df.empty:
            return []
        
        entity_titles = set(entity['title'].lower() for entity in entities)
        relevant_relationships = []
        
        for _, rel in self.relationships_df.iterrows():
            source = str(rel.get('source', '')).lower()
            target = str(rel.get('target', '')).lower()
            
            if source in entity_titles or target in entity_titles:
                relevant_relationships.append({
                    'source': rel.get('source', ''),
                    'target': rel.get('target', ''),
                    'description': rel.get('description', ''),
                    'weight': rel.get('weight', 1.0)
                })
        
        # Sort by weight and return top relationships
        relevant_relationships.sort(key=lambda x: x.get('weight', 0), reverse=True)
        return relevant_relationships[:self.max_relationships]
    
    def _find_relevant_reports(self) -> List[Dict[str, Any]]:
        """Get top community reports for general context."""
        if self.reports_df is None or self.reports_df.empty:
            return []
        
        relevant_reports = []
        for _, report in self.reports_df.head(self.max_reports).iterrows():
            relevant_reports.append({
                'title': report.get('title', ''),
                'summary': report.get('summary', ''),
                'rating': report.get('rating', 0.0),
                'rating_explanation': report.get('rating_explanation', '')
            })
        
        return relevant_reports
    
    def _build_knowledge_context(self, question: str) -> str:
        """Build knowledge context from graph data without retrieval."""
        if not self.indexed:
            return ""
        
        context_parts = []
        
        # Find relevant entities
        relevant_entities = self._find_relevant_entities(question)
        if relevant_entities:
            context_parts.append("Relevant Knowledge Entities:")
            for entity in relevant_entities:
                context_parts.append(f"- {entity['title']} ({entity['type']}): {entity['description']}")
        
        # Find relevant relationships
        relevant_relationships = self._find_relevant_relationships(relevant_entities)
        if relevant_relationships:
            context_parts.append("\nRelevant Relationships:")
            for rel in relevant_relationships:
                context_parts.append(f"- {rel['source']} ← → {rel['target']}: {rel['description']}")
        
        # Add community insights
        relevant_reports = self._find_relevant_reports()
        if relevant_reports:
            context_parts.append("\nCommunity Analysis:")
            for report in relevant_reports:
                summary = report['summary'][:300] + "..." if len(report['summary']) > 300 else report['summary']
                context_parts.append(f"- {report['title']}: {summary}")
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def _enhance_sample_with_knowledge(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance sample with knowledge graph context."""
        question = sample.get('question', '')
        
        # Build knowledge context
        knowledge_context = self._build_knowledge_context(question)
        
        # Create enhanced sample
        enhanced_sample = sample.copy()
        
        if knowledge_context:
            # Add knowledge context to existing context if any
            existing_context = sample.get('context', '')
            if existing_context:
                enhanced_sample['context'] = f"{knowledge_context}\n\nAdditional Context:\n{existing_context}"
            else:
                enhanced_sample['context'] = knowledge_context
            
            enhanced_sample['technique'] = 'graphllm_v2'
        
        return enhanced_sample
    
    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of samples with GraphLLM v2 enhancement."""
        
        # Auto-index if needed and enabled
        if not self.indexed and self.auto_index:
            logger.info("Auto-indexing samples for GraphLLM v2...")
            
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
                logger.warning("Failed to index samples, falling back to simple LLM")
                return self.llm_system.batch_process_samples(samples)
        
        results = []
        
        for sample in samples:
            try:
                # Enhance sample with knowledge graph context
                enhanced_sample = self._enhance_sample_with_knowledge(sample)
                
                # Process through LLM system
                result = self.llm_system.process_sample(enhanced_sample)
                
                # Add GraphLLM v2 metadata
                result.update({
                    'system_type': 'graphllm_v2',
                    'knowledge_enhanced': self.indexed,
                    'entities_available': len(self.entities_df) if self.entities_df is not None else 0,
                    'relationships_available': len(self.relationships_df) if self.relationships_df is not None else 0,
                    'reports_available': len(self.reports_df) if self.reports_df is not None else 0,
                })
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                # Fallback to simple LLM
                fallback_result = self.llm_system.process_sample(sample)
                fallback_result.update({
                    'system_type': 'graphllm_v2_fallback',
                    'error': str(e)
                })
                results.append(fallback_result)
        
        return results
