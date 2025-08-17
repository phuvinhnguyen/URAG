from systems.abstract import AbstractRAGSystem
from systems.simplellm import SimpleLLMSystem
from typing import Dict, Any, List
from loguru import logger


class SimpleRAGSystem(AbstractRAGSystem):
    """
    Simple RAG system that performs keyword-based retrieval and augmentation.
    
    This demonstrates how to implement a traditional RAG system with retrieval.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", device: str = "auto", **kwargs):
        """Initialize the RAG system with an LLM and simple retrieval."""
        # Initialize the LLM component
        self.llm_system = SimpleLLMSystem(model_name, device)
        
        # Simple knowledge base (in practice, this would be a vector database)
        self.knowledge_base = {
            "france": "France is a country in Europe. Its capital is Paris, which is known for landmarks like the Eiffel Tower.",
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
        
        logger.info(f"Initialized SimpleRAG with {len(self.knowledge_base)} knowledge entries")
    
    def get_batch_size(self) -> int:
        """Return batch size."""
        return 1
    
    def _retrieve_context(self, question: str) -> List[str]:
        """Simple keyword-based retrieval."""
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
        """Process sample with RAG enhancement."""
        # Retrieve relevant context
        question = sample.get('question', '')
        retrieved_docs = self._retrieve_context(question)
        
        # Augment sample with retrieved context
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
        
        # Add RAG-specific information
        result.update({
            'retrieved_docs': retrieved_docs,
            'num_retrieved_docs': len(retrieved_docs),
            'rag_enhanced': bool(retrieved_docs),
            'system_type': 'simple_rag'
        })
        
        return result
