from systems.abstract import AbstractRAGSystem
from systems.fusionllm import FusionLLMSystem
from typing import Dict, Any, List, Tuple
from loguru import logger
import re
from collections import defaultdict, Counter
import math
from utils.clean import clean_web_content  # pyright: ignore[reportMissingImports]
from utils.get_html import get_web_content  # pyright: ignore[reportMissingImports]

class FusionRAGSystem(AbstractRAGSystem):
    """
    Fusion RAG system that uses multiple diverse queries and Reciprocal Rank Fusion (RRF).
    
    This system implements the Fusion RAG technique:
    1. Generate multiple diverse queries from the original question
    2. Retrieve relevant documents for each query
    3. Apply Reciprocal Rank Fusion to combine and rank results
    4. Generate final answer using the top-ranked fused documents
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto", num_queries: int = 3, k: int = 60, **kwargs):
        """Initialize the Fusion RAG system with an LLM and enhanced retrieval."""
        # Initialize the Fusion LLM component
        self.llm_system = FusionLLMSystem(model_name, device, num_queries=num_queries)
        self.k = k  # RRF parameter (higher values reduce the impact of high-ranked documents)
        
        # Enhanced knowledge base with comprehensive information
        self.knowledge_base = {
            "france": "France is a country in Western Europe with a rich history and culture. Its capital is Paris, home to iconic landmarks like the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. France is known for its contributions to art, literature, philosophy, and cuisine. The French Revolution of 1789 was a pivotal moment in world history.",
            "capital": "A capital city serves as the primary administrative, political, and often economic center of a country or region. Capitals typically house the main government institutions, including the executive, legislative, and judicial branches. Examples include Washington D.C. for the United States, London for the United Kingdom, and Tokyo for Japan.",
            "python": "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability and simplicity, making it popular for beginners and professionals alike. Python is widely used in web development, data science, artificial intelligence, automation, and scientific computing.",
            "programming": "Programming languages are formal languages designed to communicate instructions to computers. They provide a way for humans to create software applications, websites, games, and systems. Popular programming languages include Python, JavaScript, Java, C++, and Go, each with their own strengths and use cases.",
            "data science": "Data science is an interdisciplinary field that combines statistical analysis, programming, and domain expertise to extract meaningful insights from data. It involves data collection, cleaning, exploration, modeling, and visualization. Data scientists use tools like Python, R, SQL, and machine learning algorithms to solve complex problems.",
            "jupiter": "Jupiter is the largest planet in our solar system and the fifth planet from the Sun. This gas giant has a mass greater than all other planets combined and features a distinctive Great Red Spot, a massive storm that has been raging for centuries. Jupiter has over 80 known moons, including the four largest Galilean moons: Io, Europa, Ganymede, and Callisto.",
            "planet": "Planets are celestial bodies that orbit stars and have sufficient mass to maintain a roughly spherical shape while clearing their orbital neighborhood of other objects. Our solar system contains eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune, each with unique characteristics and compositions.",
            "solar system": "Our solar system formed approximately 4.6 billion years ago from a collapsing cloud of gas and dust. It consists of the Sun, eight planets, their moons, asteroids, comets, and other debris. The solar system is located in the Milky Way galaxy and continues to evolve through gravitational interactions and cosmic events.",
            "shakespeare": "William Shakespeare (1564-1616) was an English playwright and poet widely regarded as the greatest writer in the English language. He wrote approximately 39 plays and 154 sonnets, creating timeless works like Romeo and Juliet, Hamlet, Macbeth, and A Midsummer Night's Dream. His influence on literature and the English language is immeasurable.",
            "literature": "Literature encompasses written works of artistic, intellectual, or cultural value, including novels, poetry, drama, essays, and short stories. It reflects human experiences, emotions, and ideas across different cultures and time periods. Literature serves to entertain, educate, and provide insight into the human condition.",
            "pride and prejudice": "Pride and Prejudice, published in 1813, is one of Jane Austen's most beloved novels. Set in Regency-era England, it follows the complex relationship between Elizabeth Bennet and Mr. Darcy, exploring themes of love, marriage, social class, and personal growth. The novel is celebrated for its wit, character development, and social commentary.",
            "gold": "Gold is a precious metal with the chemical symbol Au (from Latin 'aurum') and atomic number 79. Highly valued throughout history for its beauty, rarity, and resistance to corrosion, gold has been used as currency, jewelry, and in electronic components. It plays important roles in economics, technology, and cultural traditions worldwide.",
            "chemical": "Chemical elements are pure substances consisting of atoms with the same number of protons in their nuclei. The periodic table organizes all 118 known elements by atomic number and properties. Chemistry studies how elements combine to form compounds and how they interact in chemical reactions that drive biological processes and industrial applications.",
            "world war": "World War II (1939-1945) was the deadliest conflict in human history, involving most of the world's nations. It began with Germany's invasion of Poland and ended with the surrender of Japan following the atomic bombings of Hiroshima and Nagasaki. The war reshaped global politics, led to the creation of the United Nations, and marked the beginning of the Cold War era.",
            "war": "Wars are large-scale armed conflicts between nations, states, or organized groups. Throughout history, wars have been fought over territory, resources, ideology, and power. Major wars like World War I, World War II, and the Cold War have profoundly shaped modern civilization, technology, and international relations.",
            "mathematics": "Mathematics is the abstract science of numbers, quantities, shapes, and patterns. It includes branches like algebra, geometry, calculus, statistics, and number theory. Mathematics provides the foundation for physics, engineering, computer science, economics, and many other fields, serving as a universal language for describing natural phenomena.",
            "math": "Mathematical concepts and operations form the basis of quantitative reasoning and problem-solving. From basic arithmetic (addition, subtraction, multiplication, division) to advanced topics like differential equations and abstract algebra, mathematics helps us understand patterns, relationships, and structures in the world around us.",
            "brazil": "Brazil is the largest country in South America, covering nearly half of the continent's land area. Its capital is Brasília, while São Paulo and Rio de Janeiro are its largest cities. Brazil is known for the Amazon rainforest, diverse culture, Portuguese language, vibrant festivals like Carnival, and its contributions to music, sports, and literature.",
            "south america": "South America is a continent comprising 12 sovereign countries and numerous territories. It's home to the Amazon rainforest, the Andes mountains, and diverse ecosystems. The continent has a rich cultural heritage influenced by indigenous peoples, European colonization, and African heritage, with Spanish and Portuguese as the dominant languages.",
            "html": "HTML (Hypertext Markup Language) is the standard markup language for creating web pages and web applications. Developed by Tim Berners-Lee, HTML uses tags to structure content, define headings, paragraphs, links, images, and other elements. Modern HTML5 includes semantic elements and improved support for multimedia and interactive content.",
            "web": "The World Wide Web is a global information system that allows users to access and share documents and resources over the internet. Built on technologies like HTML, CSS, and JavaScript, the web has revolutionized communication, commerce, education, and entertainment, connecting billions of people worldwide.",
            "nitrogen": "Nitrogen is a chemical element with symbol N and atomic number 7. It constitutes about 78% of Earth's atmosphere and is essential for life, forming a crucial component of amino acids, proteins, and nucleic acids like DNA. The nitrogen cycle describes how nitrogen moves through ecosystems, supporting plant growth and biological processes.",
            "atmosphere": "Earth's atmosphere is a layer of gases surrounding our planet, held in place by gravity. Composed primarily of nitrogen (78%) and oxygen (21%), it protects life by filtering harmful solar radiation, regulating temperature, and enabling weather patterns. The atmosphere consists of several layers: troposphere, stratosphere, mesosphere, and thermosphere.",
            "atom": "Atoms are the fundamental building blocks of matter, consisting of a nucleus containing protons and neutrons, surrounded by electrons in orbital shells. The arrangement and number of these subatomic particles determine an element's properties and behavior in chemical reactions. Atomic theory explains the structure and interactions of matter at the smallest scale.",
            "matter": "Matter is anything that has mass and occupies space, existing in various states including solid, liquid, gas, and plasma. All matter is composed of atoms and molecules that interact through electromagnetic and nuclear forces. The study of matter and its properties forms the foundation of physics and chemistry.",
            "portuguese": "Portuguese is a Romance language spoken by over 260 million people worldwide, making it the sixth most spoken language globally. It's the official language of Portugal, Brazil, and several African countries. Portuguese evolved from Latin and shares similarities with Spanish, Italian, and French, while maintaining its unique characteristics.",
            "language": "Languages are complex systems of communication that allow humans to express thoughts, emotions, and ideas through spoken, written, or signed symbols. There are approximately 7,000 languages spoken worldwide, each reflecting unique cultural perspectives and ways of understanding the world. Language shapes thought and social identity.",
            "mona lisa": "The Mona Lisa, painted by Leonardo da Vinci between 1503-1519, is arguably the world's most famous painting. Housed in the Louvre Museum in Paris, it depicts Lisa Gherardini with an enigmatic smile that has captivated viewers for centuries. The painting showcases da Vinci's masterful use of sfumato technique and psychological depth.",
            "leonardo": "Leonardo da Vinci (1452-1519) was an Italian Renaissance polymath whose genius spanned art, science, engineering, and invention. Known for masterpieces like the Mona Lisa and The Last Supper, he also designed flying machines, studied anatomy, and made groundbreaking observations in various scientific fields. He embodies the Renaissance ideal of the universal genius.",
            "art": "Art encompasses various forms of human creative expression, including visual arts (painting, sculpture, photography), performing arts (music, dance, theater), and literary arts. Art serves to communicate emotions, ideas, and cultural values, providing beauty, meaning, and commentary on the human experience throughout history.",
            "cpu": "The Central Processing Unit (CPU) is the primary component of a computer responsible for executing instructions and performing calculations. Modern CPUs contain billions of transistors organized into cores that can process multiple tasks simultaneously. CPU performance is measured by factors like clock speed, number of cores, and architecture efficiency.",
            "computer": "Computers are electronic devices that process data according to programmed instructions. They consist of hardware components (CPU, memory, storage, input/output devices) and software (operating systems, applications). From personal computers to supercomputers, these machines have revolutionized how we work, communicate, and access information."
        }
        
        logger.info(f"Initialized Fusion RAG with {len(self.knowledge_base)} knowledge entries")
    
    def get_batch_size(self) -> int:
        """Return batch size."""
        return 1
    
    def _extract_existing_documents(self, sample: Dict[str, Any]) -> List[str]:
        """Extract and process documents from search_results field following SimpleRAG approach."""
        search_results = sample.get('search_results', [])
        documents = []
        
        if isinstance(search_results, list):
            # CRAG format: array of document objects
            for doc in search_results:
                if isinstance(doc, dict):

                    if doc.get('page_result'):
                        # Use full HTML content + snippet
                        snippet = doc.get('page_snippet', '')
                        full_content = snippet + "\n\n" + clean_web_content(doc['page_result'])
                        documents.append(full_content)
                        logger.debug(f"Using page_result + snippet (length: {len(full_content)})")
                    elif doc.get('page_url'):
                        # Fallback: fetch from URL + snippet  
                        snippet = doc.get('page_snippet', '')
                        fetched_content = get_web_content(doc['page_url'])
                        if fetched_content:
                            full_content = snippet + "\n\n" + clean_web_content(fetched_content)
                            documents.append(full_content)
                            logger.debug(f"Fetched content from page_url + snippet (length: {len(full_content)})")
                        else:
                            # Final fallback: just snippet
                            if snippet:
                                documents.append(snippet)
                                logger.debug(f"Using snippet only as fallback (length: {len(snippet)})")
                    else:
                        # Legacy support for other content fields
                        content = (doc.get('content', '') or 
                                 doc.get('text', '') or 
                                 doc.get('page_content', '') or
                                 doc.get('snippet', '') or
                                 doc.get('page_snippet', '') or
                                 doc.get('body', ''))
                        
                        # If no content field, concatenate page_name and available text
                        if not content:
                            page_name = doc.get('page_name', '')
                            title = doc.get('title', '')
                            summary = doc.get('summary', '')
                            if page_name or title or summary:
                                content = f"{title or page_name}\n{summary}".strip()
                        
                        if content:
                            documents.append(content)
                            logger.debug(f"Using legacy content field (length: {len(content)})")
                elif isinstance(doc, str):
                    # Simple string format
                    documents.append(doc)
        elif isinstance(search_results, str):
            # Simple string format
            documents.append(search_results)
        
        logger.debug(f"Extracted {len(documents)} processed documents from search_results")
        return documents

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for matching."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
        
        # Extract words, convert to lowercase, remove punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _compute_semantic_similarity(self, query: str, document: str) -> float:
        """Compute semantic similarity between query and document."""
        query_keywords = set(self._extract_keywords(query))
        doc_keywords = set(self._extract_keywords(document))
        
        if not query_keywords or not doc_keywords:
            return 0.0
        
        # Compute Jaccard similarity
        intersection = len(query_keywords.intersection(doc_keywords))
        union = len(query_keywords.union(doc_keywords))
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # Boost score for exact keyword matches in document
        exact_matches = 0
        for keyword in query_keywords:
            if keyword in document.lower():
                exact_matches += 1
        
        exact_match_bonus = exact_matches / len(query_keywords) if query_keywords else 0.0
        
        # Combine similarities with weights
        final_score = 0.6 * jaccard_sim + 0.4 * exact_match_bonus
        return final_score
    
    def _retrieve_for_single_query(self, query: str, documents: List[str], is_kb: bool = False) -> List[Tuple[int, float, str]]:
        """Retrieve and rank documents for a single query."""
        scored_docs = []
        
        if is_kb:
            # Use knowledge base
            for keyword, doc in documents:  # documents is actually knowledge_base.items()
                score = self._compute_semantic_similarity(query, doc)
                
                # Also check for direct keyword relevance
                query_lower = query.lower()
                keyword_score = 0
                if keyword in query_lower:
                    keyword_score = 2  # Exact keyword match
                else:
                    # Check for partial matches
                    keyword_words = keyword.split()
                    for word in keyword_words:
                        if word in query_lower:
                            keyword_score += 1
                
                # Combine semantic similarity with keyword relevance
                final_score = 0.7 * score + 0.3 * (keyword_score / 3.0)  # Normalize keyword score
                
                if final_score > 0.05:  # Threshold for relevance
                    scored_docs.append((hash(keyword), final_score, doc))
        else:
            # Use existing documents
            for i, doc in enumerate(documents):
                score = self._compute_semantic_similarity(query, doc)
                
                if score > 0.05:  # Threshold for relevance
                    scored_docs.append((i, score, doc))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
    
    def _apply_reciprocal_rank_fusion(self, query_results: List[List[Tuple[int, float, str]]]) -> List[Tuple[str, float]]:
        """Apply Reciprocal Rank Fusion to combine results from multiple queries."""
        # Document ID to (document, total_rrf_score)
        doc_scores = defaultdict(float)
        doc_content = {}
        
        for query_idx, results in enumerate(query_results):
            for rank, (doc_id, relevance_score, doc) in enumerate(results):
                # RRF formula: 1 / (k + rank) where k is typically 60
                rrf_score = 1.0 / (self.k + rank + 1)  # +1 because rank is 0-indexed
                doc_scores[doc_id] += rrf_score
                doc_content[doc_id] = doc
                
                logger.debug(f"Query {query_idx}, Rank {rank}: Doc {doc_id} gets RRF score {rrf_score:.4f}")
        
        # Sort by total RRF score (descending)
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return (document_content, final_rrf_score)
        fused_results = [(doc_content[doc_id], score) for doc_id, score in sorted_docs]
        
        logger.info(f"RRF combined {len(fused_results)} unique documents from {len(query_results)} query results")
        return fused_results
    
    def _retrieve_with_fusion(self, queries: List[str], existing_docs: List[str] = None) -> List[str]:
        """Retrieve documents using multiple queries and Reciprocal Rank Fusion."""
        all_query_results = []
        
        for i, query in enumerate(queries):
            if existing_docs:
                # Use existing documents from dataset
                query_results = self._retrieve_for_single_query(query, existing_docs, is_kb=False)
                logger.debug(f"Query {i}: '{query}' retrieved {len(query_results)} docs from existing documents")
            else:
                # Use built-in knowledge base
                query_results = self._retrieve_for_single_query(query, list(self.knowledge_base.items()), is_kb=True)
                logger.debug(f"Query {i}: '{query}' retrieved {len(query_results)} docs from knowledge base")
            
            all_query_results.append(query_results)
        
        # Apply Reciprocal Rank Fusion
        fused_results = self._apply_reciprocal_rank_fusion(all_query_results)
        
        # Extract top documents (limit to top 5 for context length management)
        top_docs = [doc for doc, score in fused_results[:5]]
        
        logger.info(f"Fusion retrieval returned {len(top_docs)} top-ranked documents")
        return top_docs
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample with Fusion RAG enhancement."""
        question = sample.get('question', '')
        
        # Step 1: Generate diverse queries using Fusion LLM
        diverse_queries = self.llm_system.generate_diverse_queries(question)
        logger.info(f"Generated diverse queries: {diverse_queries}")
        
        # Step 2: Extract existing documents from dataset (if available)
        existing_docs = self._extract_existing_documents(sample)
        
        # Step 3: Perform fusion retrieval
        if existing_docs:
            logger.info(f"Using {len(existing_docs)} existing documents from dataset for fusion retrieval")
            retrieved_docs = self._retrieve_with_fusion(diverse_queries, existing_docs)
            retrieval_method = 'existing_docs_fusion'
            source_type = 'dataset_provided'
        else:
            logger.info("No existing documents found, using built-in knowledge base for fusion retrieval")
            retrieved_docs = self._retrieve_with_fusion(diverse_queries)
            retrieval_method = 'knowledge_base_fusion'
            source_type = 'built_in_kb'
        
        # Step 4: Augment sample with retrieved context
        augmented_sample = sample.copy()
        if retrieved_docs:
            # Combine retrieved documents
            retrieved_context = "\n\n".join(retrieved_docs)
            combined_context = f"Context:\n{retrieved_context}"
            
            augmented_sample['search_results'] = combined_context
            augmented_sample['technique'] = 'fusion'
            
            logger.debug(f"Enhanced sample with {len(retrieved_docs)} fusion-retrieved documents from {source_type}")
        else:
            logger.debug(f"No relevant documents found for fusion retrieval from {source_type}")
            augmented_sample['technique'] = 'fusion'
        
        # Step 5: Process through Fusion LLM system
        result = self.llm_system.process_sample(augmented_sample)
        
        # Step 6: Add Fusion-specific information
        result.update({
            'retrieved_docs': retrieved_docs,
            'num_retrieved_docs': len(retrieved_docs),
            'num_available_docs': len(existing_docs) if existing_docs else len(self.knowledge_base),
            'fusion_enhanced': bool(retrieved_docs),
            'system_type': 'fusion_rag',
            'diverse_queries': diverse_queries,
            'retrieval_method': retrieval_method,
            'document_source': source_type,
            'rrf_k_parameter': self.k
        })
        
        return result
