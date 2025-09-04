from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import re
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from collections import Counter
import pandas as pd
from pathlib import Path

class GraphLLMSystem(AbstractRAGSystem):
    """
    GraphRAG-enhanced LLM system that uses knowledge graph context.
    
    This system leverages structured knowledge from GraphRAG entities and relationships
    to provide enhanced context for LLM responses, following the existing patterns.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium", 
                 device: str = "auto", 
                 num_samples: int = 20, 
                 technique: str = "graphrag",
                 graphrag_data_path: str = None):
        """Initialize the GraphRAG-enhanced LLM system."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.num_samples = num_samples
        self.technique = technique
        self.graphrag_data_path = graphrag_data_path

        logger.info(f"Loading model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load GraphRAG knowledge data
        self.entities_df = None
        self.relationships_df = None
        self.community_reports_df = None
        
        if graphrag_data_path:
            self._load_graphrag_data(graphrag_data_path)
    
    def _load_graphrag_data(self, data_path: str):
        """Load GraphRAG generated data files."""
        try:
            data_path = Path(data_path)
            
            # Load entities
            entities_file = data_path / "entities.parquet"
            if entities_file.exists():
                self.entities_df = pd.read_parquet(entities_file)
                logger.info(f"Loaded {len(self.entities_df)} entities from GraphRAG data")
            
            # Load relationships
            relationships_file = data_path / "relationships.parquet"
            if relationships_file.exists():
                self.relationships_df = pd.read_parquet(relationships_file)
                logger.info(f"Loaded {len(self.relationships_df)} relationships from GraphRAG data")
            
            # Load community reports
            reports_file = data_path / "community_reports.parquet"
            if reports_file.exists():
                self.community_reports_df = pd.read_parquet(reports_file)
                logger.info(f"Loaded {len(self.community_reports_df)} community reports from GraphRAG data")
                
        except Exception as e:
            logger.warning(f"Could not load GraphRAG data from {data_path}: {e}")
    
    def get_batch_size(self) -> int:
        """Return batch size of 1 for this simple implementation."""
        return 1
    
    def _extract_relevant_entities(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Extract relevant entities based on query similarity."""
        if self.entities_df is None:
            return []
        
        relevant_entities = []
        query_lower = query.lower()
        
        # Simple keyword matching for entity relevance
        for _, entity in self.entities_df.iterrows():
            entity_title = str(entity.get('title', '')).lower()
            entity_description = str(entity.get('description', '')).lower()
            
            # Calculate simple relevance score based on keyword overlap
            relevance_score = 0
            if entity_title in query_lower or any(word in entity_title for word in query_lower.split()):
                relevance_score += 2
            if any(word in entity_description for word in query_lower.split()):
                relevance_score += 1
                
            if relevance_score > 0:
                relevant_entities.append({
                    'title': entity.get('title', ''),
                    'description': entity.get('description', ''),
                    'relevance_score': relevance_score
                })
        
        # Sort by relevance and return top_k
        relevant_entities.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_entities[:top_k]
    
    def _extract_relevant_relationships(self, entities: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Extract relationships involving the relevant entities."""
        if self.relationships_df is None:
            return []
        
        relevant_relationships = []
        entity_titles = set(entity.lower() for entity in entities)
        
        for _, relationship in self.relationships_df.iterrows():
            source = str(relationship.get('source', '')).lower()
            target = str(relationship.get('target', '')).lower()
            
            if source in entity_titles or target in entity_titles:
                relevant_relationships.append({
                    'source': relationship.get('source', ''),
                    'target': relationship.get('target', ''),
                    'description': relationship.get('description', ''),
                    'weight': relationship.get('weight', 1.0)
                })
        
        # Sort by weight and return top_k
        relevant_relationships.sort(key=lambda x: x.get('weight', 0), reverse=True)
        return relevant_relationships[:top_k]
    
    def _build_graph_context(self, query: str) -> str:
        """Build structured context from GraphRAG knowledge graph."""
        context_parts = []
        
        # Get relevant entities
        relevant_entities = self._extract_relevant_entities(query)
        if relevant_entities:
            context_parts.append("Relevant Entities:")
            for entity in relevant_entities:
                context_parts.append(f"- {entity['title']}: {entity['description']}")
        
        # Get relevant relationships
        entity_titles = [entity['title'] for entity in relevant_entities]
        relevant_relationships = self._extract_relevant_relationships(entity_titles)
        if relevant_relationships:
            context_parts.append("\nRelevant Relationships:")
            for rel in relevant_relationships:
                context_parts.append(f"- {rel['source']} -> {rel['target']}: {rel['description']}")
        
        # Get relevant community insights
        if self.community_reports_df is not None and not self.community_reports_df.empty:
            # Simple approach: take the first few community summaries
            community_summaries = self.community_reports_df['summary'].head(2).tolist()
            if community_summaries:
                context_parts.append("\nCommunity Insights:")
                for i, summary in enumerate(community_summaries, 1):
                    context_parts.append(f"- Community {i}: {str(summary)[:200]}...")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt with GraphRAG context enhancement."""
        question = sample.get('question', '')
        technique = self.technique
        
        # Build graph-based context
        graph_context = self._build_graph_context(question)
        
        if technique == 'graphrag' and graph_context:
            prompt = f"""Based on the following knowledge graph context, please answer the question.

Knowledge Graph Context:
{graph_context}

Question: {question}

Please provide your final answer in the format <answer>X</answer> where X is your answer, incorporating the relevant knowledge from the graph context."""
        elif technique == 'cot':
            return f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format <answer>X</answer> where X is your answer."
        else:
            # Direct prompting
            return f"{question}\n\nPlease provide your final answer in the format <answer>X</answer> where X is your answer."
        
        return prompt
    
    def _generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate response from the LLM."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _find_answer_positions(self, text: str) -> Tuple[int, int]:
        """Find the start and end positions of <answer>...</answer> tags."""
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.start(1), match.end(1)
        return -1, -1
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        start_pos, end_pos = self._find_answer_positions(response)
        if start_pos != -1:
            return response[start_pos:end_pos].strip()
        return "Unknown"
    
    def _generate_multiple_responses(self, prompt: str, num_samples: int) -> List[str]:
        """Generate multiple responses for empty options case."""
        responses = []
        for i in range(num_samples):
            # Use higher temperature to create diversity
            response = self._generate_response(prompt, temperature=0.8)
            answer = self._extract_answer(response)
            if answer != "Unknown":
                responses.append(answer)
            logger.debug(f"Sample {i+1}/{num_samples}: {answer}")
        return responses
    
    def _compute_probabilities_from_samples(self, answers: List[str]) -> Tuple[Dict[str, float], List[str]]:
        """Compute probabilities from multiple answer samples."""
        if not answers:
            return {}, []
        
        # Count frequency of each answer
        answer_counts = Counter(answers)
        total_count = len(answers)
        
        # Calculate probabilities
        probabilities = {}
        unique_options = list(answer_counts.keys())
        
        for answer, count in answer_counts.items():
            probabilities[answer] = count / total_count
        
        logger.info(f"Generated {len(unique_options)} unique options from {total_count} samples: {answer_counts}")
        
        return probabilities, unique_options
    
    def _compute_option_probabilities(self, response: str, options: List[str]) -> Dict[str, float]:
        """Compute softmax probabilities for each option."""
        start_pos, end_pos = self._find_answer_positions(response + '<answer>A</answer>')
        
        if start_pos == -1:
            # If no answer format found, return uniform distribution
            uniform_prob = 1.0 / len(options)
            return {option: uniform_prob for option in options}
        
        inputs = self.tokenizer.encode(response[:start_pos], return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            logits = self.model(inputs).logits[0, -1, :]

        option_tokens = [self.tokenizer.encode(option, add_special_tokens=False)[0] for option in options]

        logits = F.softmax(logits[option_tokens], dim=0)

        return {option: logits[i].item() for i, option in enumerate(options)}
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample through the GraphRAG-enhanced LLM system."""
        # Generate prompt with graph context
        prompt = self._generate_prompt(sample)
        logger.info(f"GraphLLM Prompt used: {prompt[:500]}...")
        
        # Extract options from the sample
        options = sample.get('options', [])
        
        if not options:
            # Case 1: Empty options - generate multiple responses and compute frequency-based probabilities
            logger.info(f"Empty options detected. Generating {self.num_samples} responses...")
            
            # Generate multiple responses
            answers = self._generate_multiple_responses(prompt, self.num_samples)
            
            # Compute probabilities from frequency
            option_probabilities, generated_options = self._compute_probabilities_from_samples(answers)
            
            # Get the most frequent answer as predicted answer
            if option_probabilities:
                predicted_answer = max(option_probabilities.items(), key=lambda x: x[1])[0]
            else:
                predicted_answer = "Unknown"
            
            # Generate one final response for display
            final_response = self._generate_response(prompt, temperature=0.1)
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': final_response,
                'predicted_answer': predicted_answer,
                'option_probabilities': option_probabilities,
                'num_samples_generated': len(answers),
                'prompt_used': prompt,
                'technique': self.technique,
                'method': 'frequency_based_with_graph',
                'graph_enhanced': self.graphrag_data_path is not None
            }
        else:
            # Case 2: Options provided - use original logit-based method
            logger.info(f"Using provided options: {options}")
            
            # Generate response
            response = self._generate_response(prompt, temperature=0.1)
            
            # Compute option probabilities using logits
            option_probabilities = self._compute_option_probabilities(response, options)
            
            # Extract the predicted answer
            predicted_answer = max(option_probabilities.items(), key=lambda x: x[1])[0]
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': response,
                'predicted_answer': predicted_answer,
                'option_probabilities': option_probabilities,
                'provided_options': options,
                'prompt_used': prompt,
                'technique': self.technique,
                'method': 'logit_based_with_graph',
                'graph_enhanced': self.graphrag_data_path is not None
            }