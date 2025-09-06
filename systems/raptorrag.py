from systems.abstract import AbstractRAGSystem
from systems.raptorllm import RaptorLLMSystem
from typing import Dict, Any, List
from loguru import logger
from pathlib import Path
import sys
import os
from utils.clean import clean_web_content
from utils.get_html import get_web_content
from utils.vectordb import QdrantVectorDB

# Ensure project root is on sys.path so nested packages resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    # Try nested package layout first: repo_root/raptor/raptor/*.py
    from raptor.raptor.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
    from raptor.raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig
    from raptor.raptor.EmbeddingModels import SBertEmbeddingModel
    from raptor.raptor.SummarizationModels import GPT3TurboSummarizationModel
    from raptor.raptor.tree_structures import Tree, Node
    RAPTOR_AVAILABLE = True
    logger.info("RAPTOR components loaded successfully (raptor.raptor.*)")
except ImportError:
    try:
        # Fallback to flat layout: raptor/*.py
        from raptor.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
        from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig
        from raptor.EmbeddingModels import SBertEmbeddingModel
        from raptor.SummarizationModels import GPT3TurboSummarizationModel
        from raptor.tree_structures import Tree, Node
        RAPTOR_AVAILABLE = True
        logger.info("RAPTOR components loaded successfully (raptor.*)")
    except ImportError as e:
        logger.warning(f"RAPTOR components not available: {e}")
        RAPTOR_AVAILABLE = False


class RaptorRAGSystem(AbstractRAGSystem):
    """
    RAPTOR RAG system that uses hierarchical clustering and tree structures.
    
    This system implements the RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
    approach following the exact format of SimpleLLMSystem while integrating direct RAPTOR components.
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen3-0.6B",
                 device: str = "auto",
                 num_samples: int = 20,
                 tree_save_path: str = "raptor_tree",
                 num_layers: int = 5,
                 max_tokens: int = 100,
                 threshold: float = 0.5,
                 top_k: int = 5,
                 reduction_dimension: int = 10):
        """Initialize the RAPTOR RAG system following SimpleLLMSystem format."""
        
        # Initialize the RAPTOR LLM component
        self.llm_system = RaptorLLMSystem(
            model_name=model_name,
            device=device,
            num_samples=num_samples,
            tree_save_path=str(Path(tree_save_path)),
            num_layers=num_layers,
            max_tokens=max_tokens,
            threshold=threshold,
            top_k=top_k,
        )
        
        # RAPTOR-specific parameters
        self.tree_save_path = Path(tree_save_path)
        self.tree_save_path.mkdir(exist_ok=True)
        self.num_layers = num_layers
        self.max_tokens = max_tokens
        self.threshold = threshold
        self.top_k = top_k
        self.reduction_dimension = reduction_dimension
        
        # RAPTOR components
        self.tree = None
        self.tree_retriever = None
        self.tree_builder = None
        
        if RAPTOR_AVAILABLE:
            self._initialize_raptor_components()
        else:
            logger.warning("RAPTOR not available, will use vector DB fallback")
        
        logger.info("Initialized RaptorRAG without persistent knowledge base; expecting per-sample documents")

    def _initialize_raptor_components(self):
        """Initialize RAPTOR tree builder and retriever configurations."""
        try:
            # Configure the tree builder
            self.tree_config = ClusterTreeConfig(
                max_tokens=self.max_tokens,
                num_layers=self.num_layers,
                threshold=self.threshold,
                top_k=self.top_k,
                selection_mode="top_k",
                summarization_length=100,
                summarization_model=GPT3TurboSummarizationModel(),
                embedding_models={"SBert": SBertEmbeddingModel()},
                cluster_embedding_model="SBert",
                reduction_dimension=self.reduction_dimension
            )
            
            # Initialize the tree builder
            self.tree_builder = ClusterTreeBuilder(self.tree_config)
            
            logger.info("RAPTOR components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAPTOR components: {e}")
            self.tree_builder = None
    
    def _build_raptor_tree(self):
        """Deprecated: no longer builds from a static knowledge base."""
        logger.debug("_build_raptor_tree called but static KB has been removed; skipping.")

    def _retrieve_context_raptor_from_documents(self, question: str, documents: List[str]) -> List[str]:
        """Build a temporary RAPTOR tree from provided documents and retrieve context."""
        if not RAPTOR_AVAILABLE or not self.tree_builder:
            return []
        try:
            combined_text = "\n\n".join(documents)
            temp_tree = self.tree_builder.build_from_text(combined_text)

            retriever_layers = min(self.num_layers, temp_tree.num_layers + 1)
            retriever_config = TreeRetrieverConfig(
                threshold=self.threshold,
                top_k=self.top_k,
                selection_mode="top_k",
                context_embedding_model="SBert",
                embedding_model=SBertEmbeddingModel(),
                num_layers=retriever_layers
            )
            temp_retriever = TreeRetriever(retriever_config, temp_tree)

            context = temp_retriever.retrieve(
                query=question,
                top_k=self.top_k,
                max_tokens=1000,
                collapse_tree=True
            )

            if not context:
                return []

            context_chunks = []
            current_chunk = ""
            sentences = context.split('. ')
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 200:
                    current_chunk += sentence + '. '
                else:
                    if current_chunk:
                        context_chunks.append(current_chunk.strip())
                    current_chunk = sentence + '. '
            if current_chunk:
                context_chunks.append(current_chunk.strip())
            return context_chunks[:3]
        except Exception as e:
            logger.error(f"RAPTOR per-doc retrieval failed: {str(e)}")
            return []
    
    def _save_tree(self):
        """Save the current RAPTOR tree."""
        if self.tree:
            try:
                tree_file = self.tree_save_path / "raptor_tree.pkl"
                import pickle
                with open(tree_file, 'wb') as f:
                    pickle.dump(self.tree, f)
                logger.info(f"RAPTOR tree saved to {tree_file}")
            except Exception as e:
                logger.error(f"Failed to save tree: {e}")
    
    def _load_tree(self):
        """Load existing RAPTOR tree."""
        tree_file = self.tree_save_path / "raptor_tree.pkl"
        if tree_file.exists():
            try:
                import pickle
                with open(tree_file, 'rb') as f:
                    self.tree = pickle.load(f)
                    
                # Reinitialize retriever with loaded tree
                retriever_layers = min(self.num_layers, self.tree.num_layers + 1)
                retriever_config = TreeRetrieverConfig(
                    threshold=self.threshold,
                    top_k=self.top_k,
                    selection_mode="top_k",
                    context_embedding_model="SBert",
                    embedding_model=SBertEmbeddingModel(),
                    num_layers=retriever_layers
                )
                self.tree_retriever = TreeRetriever(retriever_config, self.tree)
                
                logger.info(f"Successfully loaded RAPTOR tree with {len(self.tree.all_nodes)} nodes")
                return True
            except Exception as e:
                logger.error(f"Failed to load tree: {e}")
                return False
        return False

    def get_batch_size(self) -> int:
        """Return batch size (identical to SimpleLLMSystem)."""
        return 1
    
    # Removed simple keyword-based retrieval paths; retrieval is per-sample via RAPTOR or vector DB

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with RAPTOR RAG using per-sample documents or vector DB fallback."""
        question = sample.get('question', '')
        # Build documents from search results if provided
        documents: List[str] = []
        try:
            raw_docs = sample.get('search_results', [])
            if isinstance(raw_docs, list):
                for d in raw_docs:
                    if isinstance(d, dict):
                        snippet = d.get('page_snippet', '') or ''
                        page_result = d.get('page_result')
                        page_url = d.get('page_url')
                        content = ''
                        if page_result:
                            content = clean_web_content(page_result)
                        elif page_url:
                            try:
                                content = clean_web_content(get_web_content(page_url))
                            except Exception as e:
                                logger.debug(f"Failed to fetch page_url content: {e}")
                                content = ''
                        doc_text = (snippet + "\n\n" + content).strip()
                        if doc_text:
                            documents.append(doc_text)
        except Exception as e:
            logger.warning(f"Failed to build documents from search_results: {e}")

        retrieved_docs: List[str] = []
        if documents and RAPTOR_AVAILABLE and self.tree_builder:
            retrieved_docs = self._retrieve_context_raptor_from_documents(question, documents)

        # Vector DB fallback or primary when RAPTOR unavailable
        if not retrieved_docs and documents:
            try:
                database = QdrantVectorDB(
                    texts=documents,
                    embedding_model="sentence_transformers",
                    chunk_size=300,
                    overlap=150
                )
                vector_hits = database.search(question, method="hybrid", k=3)
                retrieved_docs = [hit['chunk'] for hit in vector_hits]
            except Exception as e:
                logger.warning(f"Vector DB retrieval failed: {e}")
                retrieved_docs = []
        
        # Augment sample with retrieved context (identical to SimpleRAGSystem)
        augmented_sample = sample.copy()
        if retrieved_docs:
            # Combine retrieved documents
            retrieved_context = "\n".join(retrieved_docs)
            
            # Add to existing context if any
            existing_context = sample.get('search_results', sample.get('context', ''))
            if existing_context:
                combined_context = f"{existing_context}\n\nAdditional context:\n{retrieved_context}"
            else:
                combined_context = f"Context:\n{retrieved_context}"
            
            augmented_sample['search_results'] = combined_context
            augmented_sample['technique'] = 'rag'
            
            logger.debug(f"Enhanced sample with {len(retrieved_docs)} retrieved documents")
        else:
            logger.debug("No relevant documents found for retrieval")
        
        # Process through LLM
        result = self.llm_system.process_sample(augmented_sample)
        
        # Add RAPTOR-specific information
        result.update({
            'retrieved_docs': retrieved_docs,
            'num_retrieved_docs': len(retrieved_docs),
            'rag_enhanced': bool(retrieved_docs),
            'system_type': 'raptor_rag',
            'raptor_available': RAPTOR_AVAILABLE,
            'tree_layers': 0,
            'tree_nodes': 0
        })
        
        return result
    
    def get_tree_info(self) -> Dict[str, Any]:
        """Get information about the current RAPTOR tree."""
        if not self.tree:
            return {
                'tree_available': False,
                'raptor_available': RAPTOR_AVAILABLE,
                'error': 'Tree not built or RAPTOR not available'
            }
        
        return {
            'tree_available': True,
            'raptor_available': RAPTOR_AVAILABLE,
            'num_layers': self.tree.num_layers,
            'total_nodes': len(self.tree.all_nodes),
            'leaf_nodes': len(self.tree.leaf_nodes),
            'root_nodes': len(self.tree.root_nodes),
            'layer_distribution': {
                layer: len(nodes) for layer, nodes in self.tree.layer_to_nodes.items()
            }
        }
