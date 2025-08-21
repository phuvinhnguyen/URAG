from systems.abstract import AbstractRAGSystem
from systems.hydellm import HyDELLMSystem
from typing import Dict, Any, List
from loguru import logger
import re


class HyDERAGSystem(AbstractRAGSystem):
    """
    HyDE RAG system that uses hypothetical document embeddings for retrieval.
    
    This system implements the HyDE (Hypothetical Document Embeddings) technique:
    1. Generate a hypothetical document that would answer the query
    2. Use that hypothetical document for retrieval instead of the original query
    3. Retrieve relevant documents based on hypothetical document similarity
    4. Generate final answer using retrieved context
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto", **kwargs):
        """Initialize the HyDE RAG system with an LLM and enhanced retrieval."""
        # Initialize the HyDE LLM component
        self.llm_system = HyDELLMSystem(model_name, device)
        
        # Enhanced knowledge base with more comprehensive information
        # In practice, this would be a vector database with embeddings
        self.knowledge_base = {
            "france": "France is a country in Western Europe. Its capital is Paris, which is known for landmarks like the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. France has a rich history dating back to ancient times and has been a major cultural and political force in Europe.",
            "capital": "A capital city is the primary city of a country or region, usually where the government is located. Capitals often serve as the political, economic, and cultural centers of their respective countries.",
            "python": "Python is a high-level, interpreted programming language widely used in data science, web development, automation, and artificial intelligence. It was created by Guido van Rossum and first released in 1991. Python is known for its simple, readable syntax and extensive library ecosystem.",
            "programming": "Programming languages are formal languages used to communicate instructions to computers. They allow developers to create software applications, websites, and systems by writing code that computers can execute.",
            "data science": "Data science is an interdisciplinary field that combines statistics, programming, and domain expertise to extract insights and knowledge from data. It involves data collection, cleaning, analysis, and visualization to solve complex problems.",
            "jupiter": "Jupiter is the largest planet in our solar system, a gas giant with a mass greater than all other planets combined. It has over 70 known moons, including the four largest called the Galilean moons: Io, Europa, Ganymede, and Callisto.",
            "planet": "Planets are celestial bodies that orbit stars and have sufficient mass to maintain a roughly spherical shape. In our solar system, there are eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
            "solar system": "Our solar system consists of the Sun and all celestial objects that orbit it, including eight planets, their moons, asteroids, comets, and other debris. It formed approximately 4.6 billion years ago from a collapsing cloud of gas and dust.",
            "shakespeare": "William Shakespeare (1564-1616) was an English playwright and poet, widely regarded as the greatest writer in the English language. He wrote approximately 39 plays and 154 sonnets, including famous works like Romeo and Juliet, Hamlet, and Macbeth.",
            "literature": "Literature encompasses written works of artistic or intellectual value, including novels, poetry, drama, and essays. It reflects human experiences, cultures, and ideas across different time periods and societies.",
            "pride and prejudice": "Pride and Prejudice is a novel by Jane Austen published in 1813. It follows Elizabeth Bennet and her relationship with the wealthy Mr. Darcy, exploring themes of love, marriage, social class, and personal growth in Regency-era England.",
            "gold": "Gold is a precious metal with the chemical symbol Au (from Latin 'aurum'). It is highly valued for its rarity, beauty, and resistance to corrosion. Gold has been used as currency, jewelry, and in electronics for thousands of years.",
            "chemical": "Chemical elements are pure substances consisting of atoms with the same number of protons in their nuclei. The periodic table organizes all known elements by their atomic number and properties.",
            "world war": "World War II (1939-1945) was a global conflict involving most of the world's nations. It resulted in significant changes to world politics, technology, and society, ending with the defeat of the Axis powers by the Allied forces.",
            "war": "Wars are large-scale conflicts between nations, states, or groups. Throughout history, major wars have shaped civilizations, borders, and political systems, often leading to significant social and technological changes.",
            "mathematics": "Mathematics is the science of numbers, quantities, shapes, and patterns. It includes various branches such as algebra, geometry, calculus, and statistics, and serves as the foundation for many other fields including physics, engineering, and computer science.",
            "math": "Mathematical operations form the basis of computation and problem-solving. Basic operations include addition, subtraction, multiplication, and division, while advanced mathematics involves complex analysis, differential equations, and abstract algebra.",
            "brazil": "Brazil is the largest country in South America, covering about half of the continent's land area. Its capital is Brasília, and its largest city is São Paulo. Portuguese is the official language, and Brazil is known for its diverse culture, Amazon rainforest, and vibrant cities like Rio de Janeiro.",
            "south america": "South America is a continent containing 12 countries, including Brazil, Argentina, Colombia, Peru, and Chile. It is known for its diverse geography, from the Amazon rainforest to the Andes mountains, and rich cultural heritage.",
            "html": "HTML (Hypertext Markup Language) is the standard markup language for creating web pages and web applications. It describes the structure and content of web documents using elements and tags.",
            "web": "Web technologies enable the creation, display, and interaction with content on the World Wide Web. This includes HTML for structure, CSS for styling, and JavaScript for interactivity.",
            "nitrogen": "Nitrogen is a chemical element with the symbol N and atomic number 7. It makes up about 78% of Earth's atmosphere and is essential for life, forming a key component of amino acids, proteins, and nucleic acids.",
            "atmosphere": "Earth's atmosphere is a layer of gases surrounding the planet, composed primarily of nitrogen (78%) and oxygen (21%). It protects life by absorbing harmful radiation and maintaining suitable temperatures.",
            "atom": "Atoms are the fundamental building blocks of matter, consisting of a nucleus containing protons and neutrons, surrounded by electrons. The arrangement of these particles determines the element's properties.",
            "matter": "Matter is anything that has mass and occupies space. It exists in various states including solid, liquid, gas, and plasma, and is composed of atoms and molecules.",
            "portuguese": "Portuguese is a Romance language spoken by over 260 million people worldwide. It is the official language of Portugal, Brazil, and several African countries. It evolved from Latin and shares similarities with Spanish and Italian.",
            "language": "Languages are complex systems of communication used by humans to express thoughts, emotions, and ideas. They consist of vocabulary, grammar rules, and cultural contexts that vary across different societies.",
            "mona lisa": "The Mona Lisa is a famous oil painting by Leonardo da Vinci, created between 1503-1519. It depicts Lisa Gherardini and is renowned for her enigmatic smile. The painting is housed in the Louvre Museum in Paris.",
            "leonardo": "Leonardo da Vinci (1452-1519) was an Italian Renaissance polymath known for his contributions to art, science, engineering, and invention. He created masterpieces like the Mona Lisa and The Last Supper, and designed innovative machines.",
            "art": "Art encompasses various forms of creative human expression, including visual arts (painting, sculpture), performing arts (music, dance, theater), and literary arts. It reflects cultural values and individual creativity.",
            "cpu": "The CPU (Central Processing Unit) is the primary component of a computer that executes instructions and performs calculations. It consists of an arithmetic logic unit (ALU), control unit, and registers.",
            "computer": "Computers are electronic devices that process data according to programmed instructions. They consist of hardware components (CPU, memory, storage) and software (operating systems, applications) that work together to perform tasks."
        }
        
        logger.info(f"Initialized HyDE RAG with {len(self.knowledge_base)} knowledge entries")
    
    def get_batch_size(self) -> int:
        """Return batch size."""
        return 1
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for matching."""
        # Simple keyword extraction - remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
        
        # Extract words, convert to lowercase, remove punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _compute_semantic_similarity(self, hypothetical_doc: str, knowledge_text: str) -> float:
        """Compute semantic similarity between hypothetical document and knowledge text."""
        # Extract keywords from both texts
        hyp_keywords = set(self._extract_keywords(hypothetical_doc))
        know_keywords = set(self._extract_keywords(knowledge_text))
        
        if not hyp_keywords or not know_keywords:
            return 0.0
        
        # Compute Jaccard similarity
        intersection = len(hyp_keywords.intersection(know_keywords))
        union = len(hyp_keywords.union(know_keywords))
        
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # Boost score for exact keyword matches in knowledge text
        exact_matches = 0
        for keyword in hyp_keywords:
            if keyword in knowledge_text.lower():
                exact_matches += 1
        
        exact_match_bonus = exact_matches / len(hyp_keywords) if hyp_keywords else 0.0
        
        # Combine similarities with weights
        final_score = 0.6 * jaccard_sim + 0.4 * exact_match_bonus
        
        return final_score
    
    def _retrieve_context_hyde(self, question: str, hypothetical_doc: str) -> List[str]:
        """HyDE-enhanced retrieval using hypothetical document."""
        retrieved_docs = []
        
        # Score each knowledge entry against the hypothetical document
        scored_docs = []
        for keyword, doc in self.knowledge_base.items():
            # Compute similarity with hypothetical document
            hyde_score = self._compute_semantic_similarity(hypothetical_doc, doc)
            
            # Also check for direct question-keyword relevance (fallback)
            question_lower = question.lower()
            keyword_score = 0
            if keyword in question_lower:
                keyword_score = 2  # Exact keyword match
            else:
                # Check for partial matches
                keyword_words = keyword.split()
                for word in keyword_words:
                    if word in question_lower:
                        keyword_score += 1
            
            # Combine HyDE score with keyword relevance
            final_score = 0.7 * hyde_score + 0.3 * (keyword_score / 3.0)  # Normalize keyword score
            
            if final_score > 0.1:  # Threshold for relevance
                scored_docs.append((final_score, keyword, doc, hyde_score, keyword_score))
        
        # Sort by score and take top results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top 3 documents
        for final_score, keyword, doc, hyde_score, keyword_score in scored_docs[:3]:
            retrieved_docs.append(doc)
            logger.debug(f"Retrieved (final_score={final_score:.3f}, hyde={hyde_score:.3f}, keyword={keyword_score}): {keyword} -> {doc[:50]}...")
        
        return retrieved_docs
    
    def _retrieve_context_traditional(self, question: str) -> List[str]:
        """Traditional keyword-based retrieval (fallback)."""
        question_lower = question.lower()
        retrieved_docs = []
        
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
            logger.debug(f"Traditional retrieval (score={score}): {keyword} -> {doc[:50]}...")
        
        return retrieved_docs
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with HyDE RAG enhancement."""
        question = sample.get('question', '')
        
        # Step 1: Generate hypothetical document using HyDE LLM
        hypothetical_doc = self.llm_system.generate_hypothetical_document(question)
        logger.info(f"Generated hypothetical document: {hypothetical_doc[:100]}...")
        
        # Step 2: Retrieve relevant context using HyDE technique
        hyde_retrieved_docs = self._retrieve_context_hyde(question, hypothetical_doc)
        
        # Fallback to traditional retrieval if HyDE retrieval fails
        if not hyde_retrieved_docs:
            logger.info("HyDE retrieval failed, falling back to traditional retrieval")
            hyde_retrieved_docs = self._retrieve_context_traditional(question)
        
        # Step 3: Augment sample with retrieved context
        augmented_sample = sample.copy()
        if hyde_retrieved_docs:
            # Combine retrieved documents
            retrieved_context = "\n".join(hyde_retrieved_docs)
            
            # Add to existing context if any
            existing_context = sample.get('search_results', sample.get('context', ''))
            if existing_context:
                combined_context = f"{existing_context}\n\nHyDE Retrieved context:\n{retrieved_context}"
            else:
                combined_context = f"Context:\n{retrieved_context}"
            
            augmented_sample['search_results'] = combined_context
            augmented_sample['technique'] = 'hyde'
            
            logger.debug(f"Enhanced sample with {len(hyde_retrieved_docs)} HyDE-retrieved documents")
        else:
            logger.debug("No relevant documents found for HyDE retrieval")
            augmented_sample['technique'] = 'hyde'
        
        # Step 4: Process through HyDE LLM system
        result = self.llm_system.process_sample(augmented_sample)
        
        # Step 5: Add HyDE-specific information
        result.update({
            'retrieved_docs': hyde_retrieved_docs,
            'num_retrieved_docs': len(hyde_retrieved_docs),
            'hyde_enhanced': bool(hyde_retrieved_docs),
            'system_type': 'hyde_rag',
            'hypothetical_document': hypothetical_doc,
            'retrieval_method': 'hyde_semantic' if hyde_retrieved_docs else 'traditional_fallback'
        })
        
        return result
