from systems.abstract import AbstractRAGSystem
from systems.simplellm import SimpleLLMSystem
from typing import Dict, Any, List
from loguru import logger
from pathlib import Path
import sys
import os

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
        
        # Initialize the LLM component (identical to SimpleLLMSystem)
        self.llm_system = SimpleLLMSystem(model_name, device, num_samples)
        
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
            logger.warning("RAPTOR not available, falling back to simple keyword retrieval")
        
        # Knowledge base for demonstration (following SimpleRAGSystem pattern)
        self.knowledge_base = {
            "france": "France is a country in Europe. Its capital is Paris, which is known for landmarks like the Eiffel Tower and the Louvre Museum.",
            "capital": "A capital city is the primary city of a country or region, usually where the government is located.",
            "python": "Python is a popular programming language widely used in data science, web development, and automation.",
            "programming": "Programming languages are formal languages used to communicate instructions to computers.",
            "data science": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
            "jupiter": "Jupiter is the largest planet in our solar system, a gas giant with over 70 moons.",
            "planet": "Planets are celestial bodies that orbit stars and have cleared their orbital neighborhood.",
            "solar system": "Our solar system contains the Sun and eight planets, along with moons, asteroids, and comets.",
            "shakespeare": "William Shakespeare was an English playwright and poet, considered one of the greatest writers in the English language.",
            "literature": "Literature encompasses written works, especially those considered to have artistic or intellectual value.",
            "pride and prejudice": "Pride and Prejudice is a novel by Jane Austen, published in 1813, exploring themes of love and social class.",
            "gold": "Gold is a precious metal with the chemical symbol Au, valued for its rarity and resistance to corrosion.",
            "chemical": "Chemical elements are pure substances consisting of atoms with the same number of protons.",
            "world war": "World War II was a global conflict from 1939 to 1945, ending with the defeat of the Axis powers.",
            "war": "Major wars have shaped world history, involving conflicts between nations or groups.",
            "mathematics": "Mathematics is the science of numbers, quantities, and shapes, fundamental to many fields.",
            "math": "Mathematical operations include addition, subtraction, multiplication, and division.",
            "brazil": "Brazil is the largest country in South America, with Portuguese as its official language.",
            "south america": "South America is a continent containing countries like Brazil, Argentina, and Colombia.",
            "html": "HTML (Hypertext Markup Language) is the standard markup language for creating web pages.",
            "web": "Web technologies enable the creation and display of content on the World Wide Web.",
            "nitrogen": "Nitrogen makes up about 78% of Earth's atmosphere and is essential for life.",
            "atmosphere": "Earth's atmosphere is composed primarily of nitrogen and oxygen, protecting life on the planet.",
            "atom": "Atoms are the basic building blocks of matter, consisting of protons, neutrons, and electrons.",
            "matter": "Matter is anything that has mass and takes up space, existing in various states.",
            "portuguese": "Portuguese is a Romance language spoken by over 250 million people worldwide.",
            "language": "Languages are systems of communication used by humans to express thoughts and ideas.",
            "mona lisa": "The Mona Lisa is a famous painting by Leonardo da Vinci, housed in the Louvre Museum.",
            "leonardo": "Leonardo da Vinci was an Italian Renaissance artist, inventor, and scientist.",
            "art": "Art encompasses various forms of creative expression, including painting, sculpture, and music.",
            "cpu": "The CPU (Central Processing Unit) is the primary component of a computer that executes instructions.",
            "computer": "Computers are electronic devices that process data according to programmed instructions."
        }
        
        # Build RAPTOR tree if available
        if RAPTOR_AVAILABLE:
            self._build_raptor_tree()
        
        logger.info(f"Initialized RaptorRAG with {len(self.knowledge_base)} knowledge entries")

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
        """Build the RAPTOR tree from the knowledge base."""
        if not self.tree_builder:
            logger.warning("Tree builder not available")
            return
        
        try:
            # Combine all knowledge base entries into a single text corpus
            combined_text = "\n\n".join([f"{key}: {value}" for key, value in self.knowledge_base.items()])
            
            # Build the tree
            logger.info("Building RAPTOR tree from knowledge base...")
            self.tree = self.tree_builder.build_from_text(combined_text)
            
            # Configure and initialize the retriever
            # Cap retriever layers to satisfy: retriever.num_layers <= tree.num_layers + 1
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
            
            logger.info(f"Successfully built RAPTOR tree with {len(self.tree.all_nodes)} total nodes")
            logger.info(f"Tree has {self.tree.num_layers} layers")
            
            # Save the tree
            self._save_tree()
            
        except Exception as e:
            logger.error(f"Failed to build RAPTOR tree: {str(e)}")
            self.tree = None
            self.tree_retriever = None
    
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
    
    def _retrieve_context(self, question: str) -> List[str]:
        """Retrieve context using RAPTOR tree traversal or fallback to simple retrieval."""
        if self.tree_retriever:
            return self._retrieve_context_raptor(question)
        else:
            return self._retrieve_context_simple(question)
    
    def _retrieve_context_raptor(self, question: str) -> List[str]:
        """Retrieve context using RAPTOR tree traversal."""
        try:
            # Use RAPTOR tree retrieval
            context = self.tree_retriever.retrieve(
                query=question,
                top_k=self.top_k,
                max_tokens=1000,
                collapse_tree=True  # Use collapsed tree for better coverage
            )
            
            if context:
                # Split context into reasonable chunks (following SimpleRAG pattern)
                context_chunks = []
                current_chunk = ""
                sentences = context.split('. ')
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 200:  # Reasonable chunk size
                        current_chunk += sentence + '. '
                    else:
                        if current_chunk:
                            context_chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
                
                if current_chunk:
                    context_chunks.append(current_chunk.strip())
                
                logger.debug(f"Retrieved {len(context_chunks)} context chunks using RAPTOR")
                return context_chunks[:3]  # Return top 3 chunks like SimpleRAG
            else:
                logger.warning("No context retrieved from RAPTOR tree")
                return []
                
        except Exception as e:
            logger.error(f"RAPTOR retrieval failed: {str(e)}")
            return self._retrieve_context_simple(question)
    
    def _retrieve_context_simple(self, question: str) -> List[str]:
        """Simple keyword-based retrieval (identical to SimpleRAGSystem)."""
        question_lower = question.lower()
        retrieved_docs = []
        
        # Score each knowledge entry by keyword overlap
        scored_docs = []
        for keyword, doc in self.knowledge_base.items():
            score = 0
            if keyword in question_lower:
                score = 2  # Exact keyword match
            else:
                # Check for partial matches
                keyword_words = keyword.split()
                for word in keyword_words:
                    if word in question_lower:
                        score += 1
                        
            if score > 0:
                scored_docs.append((score, keyword, doc))
        
        # Sort by score and take top results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top 3 documents
        for score, keyword, doc in scored_docs[:3]:
            retrieved_docs.append(doc)
            logger.debug(f"Retrieved (score={score}): {keyword} -> {doc[:50]}...")
        
        return retrieved_docs

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with RAPTOR RAG enhancement (following SimpleLLMSystem format exactly)."""
        # Retrieve relevant context using RAPTOR or simple retrieval
        question = sample.get('question', '')
        retrieved_docs = self._retrieve_context(question)
        
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
        
        # Process through LLM (identical to SimpleRAGSystem pattern)
        result = self.llm_system.process_sample(augmented_sample)
        
        # Add RAPTOR-specific information
        result.update({
            'retrieved_docs': retrieved_docs,
            'num_retrieved_docs': len(retrieved_docs),
            'rag_enhanced': bool(retrieved_docs),
            'system_type': 'raptor_rag',
            'raptor_available': RAPTOR_AVAILABLE and self.tree is not None,
            'tree_layers': self.tree.num_layers if self.tree else 0,
            'tree_nodes': len(self.tree.all_nodes) if self.tree else 0
        })
        
        return result
    
    def add_documents(self, documents: List[str]):
        """Add new documents to the knowledge base and rebuild the tree."""
        # Add to knowledge base
        for i, doc in enumerate(documents):
            key = f"doc_{len(self.knowledge_base) + i}"
            self.knowledge_base[key] = doc
        
        logger.info(f"Added {len(documents)} new documents. Rebuilding RAPTOR tree...")
        
        # Rebuild tree if RAPTOR is available
        if RAPTOR_AVAILABLE:
            self._build_raptor_tree()
    
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
