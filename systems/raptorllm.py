from systems.abstract import AbstractRAGSystem
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import re
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from collections import Counter
import os
import pickle
from pathlib import Path

# Import RAPTOR components
from raptor.raptor import (
    RetrievalAugmentation, 
    RetrievalAugmentationConfig,
    BaseQAModel, 
    BaseEmbeddingModel, 
    BaseSummarizationModel
)

class LocalQAModel(BaseQAModel):
    """Local QA model using transformers for question answering."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Loading QA model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def answer_question(self, context: str, question: str, max_tokens: int = 150) -> str:
        """Answer question using context and question."""
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

class LocalSummarizationModel(BaseSummarizationModel):
    """Local summarization model using transformers."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Loading summarization model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        """Summarize the given context."""
        prompt = f"Summarize the following text in {max_tokens} words or less:\n\n{context}\n\nSummary:"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        summary = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return summary.strip()

class LocalEmbeddingModel(BaseEmbeddingModel):
    """Local embedding model using sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback embedding")
            self.model = None
    
    def create_embedding(self, text: str):
        """Create embedding for the given text."""
        if self.model is None:
            # Fallback: simple hash-based embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            return [float(int(hash_obj.hexdigest()[:8], 16)) / 1e8]
        
        return self.model.encode(text)

class RaptorLLMSystem(AbstractRAGSystem):
    """
    RAPTOR LLM system that uses tree-based retrieval and summarization.
    
    This system implements the RAPTOR approach: Recursive Abstractive Processing 
    for Tree-Organized Retrieval, which builds a hierarchical tree structure
    from documents for efficient information retrieval.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium", 
                 device: str = "auto", 
                 num_samples: int = 20,
                 tree_save_path: str = "raptor_tree",
                 num_layers: int = 5,
                 max_tokens: int = 100,
                 threshold: float = 0.5,
                 top_k: int = 5):
        """Initialize the RAPTOR LLM system."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.num_samples = num_samples
        self.tree_save_path = Path(tree_save_path)
        self.tree_save_path.mkdir(exist_ok=True)
        
        # Initialize local models
        self.qa_model = LocalQAModel(model_name, device)
        self.summarization_model = LocalSummarizationModel(model_name, device)
        self.embedding_model = LocalEmbeddingModel()
        
        # Initialize main LLM for response generation
        logger.info(f"Loading main LLM model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize RAPTOR configuration
        self.raptor_config = RetrievalAugmentationConfig(
            qa_model=self.qa_model,
            summarization_model=self.summarization_model,
            embedding_model=self.embedding_model,
            tb_num_layers=num_layers,
            tb_max_tokens=max_tokens,
            tb_threshold=threshold,
            tb_top_k=top_k,
            tr_threshold=threshold,
            tr_top_k=top_k
        )
        
        # Initialize RAPTOR system
        self.raptor = None
        self._load_or_create_tree()
        
        logger.info(f"RAPTOR LLM system initialized with {num_layers} layers, threshold {threshold}, top_k {top_k}")
    
    def _load_or_create_tree(self):
        """Load existing tree or create new one."""
        tree_file = self.tree_save_path / "tree.pkl"
        
        if tree_file.exists():
            try:
                logger.info(f"Loading existing RAPTOR tree from {tree_file}")
                self.raptor = RetrievalAugmentation(config=self.raptor_config, tree=str(tree_file))
                logger.info("Successfully loaded existing RAPTOR tree")
            except Exception as e:
                logger.warning(f"Failed to load existing tree: {e}")
                self.raptor = RetrievalAugmentation(config=self.raptor_config)
        else:
            logger.info("Creating new RAPTOR tree")
            self.raptor = RetrievalAugmentation(config=self.raptor_config)
    
    def _save_tree(self):
        """Save the current tree."""
        if self.raptor and self.raptor.tree:
            tree_file = self.tree_save_path / "tree.pkl"
            try:
                self.raptor.save(str(tree_file))
                logger.info(f"RAPTOR tree saved to {tree_file}")
            except Exception as e:
                logger.error(f"Failed to save tree: {e}")
    
    def get_batch_size(self) -> int:
        """Return batch size of 1 for this implementation."""
        return 1
    
    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt based on sample technique."""
        question = sample.get('question', '')
        technique = sample.get('technique', 'direct')
        
        if technique == 'cot':
            return f"Let's think step by step.\n\n{question}\n\nPlease provide your reasoning and then give your final answer in the format <answer>X</answer> where X is your answer."
        elif technique == 'rag':
            # For RAPTOR, we'll use the retrieved context
            context = sample.get('search_results', sample.get('context', ''))
            if context:
                return f"Context information: {context}\n\nQuestion: {question}\n\nPlease provide your final answer in the format <answer>X</answer>."
            else:
                return f"{question}\n\nPlease provide your final answer in the format <answer>X</answer> where X is your answer."
        else:
            # Direct prompting
            return f"{question}\n\nPlease provide your final answer in the format <answer>X</answer> where X is your answer."
    
    def _add_documents_to_raptor(self, sample: Dict[str, Any]):
        """Add documents from sample to RAPTOR tree."""
        context = sample.get('search_results', sample.get('context', ''))
        if context and isinstance(context, str) and len(context.strip()) > 0:
            try:
                logger.info("Adding documents to RAPTOR tree")
                self.raptor.add_documents(context)
                self._save_tree()
                logger.info("Successfully added documents to RAPTOR tree")
            except Exception as e:
                logger.error(f"Failed to add documents to RAPTOR tree: {e}")
    
    def _retrieve_from_raptor(self, question: str) -> str:
        """Retrieve relevant context from RAPTOR tree."""
        try:
            if self.raptor and self.raptor.retriever:
                logger.info("Retrieving context from RAPTOR tree")
                context = self.raptor.retrieve(
                    question=question,
                    top_k=5,
                    max_tokens=2000,
                    collapse_tree=True,
                    return_layer_information=False
                )
                logger.info(f"Retrieved context length: {len(context)}")
                return context
            else:
                logger.warning("RAPTOR retriever not available")
                return ""
        except Exception as e:
            logger.error(f"Failed to retrieve from RAPTOR: {e}")
            return ""
    
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
        
        answer_counts = Counter(answers)
        total_count = len(answers)
        
        probabilities = {}
        unique_options = list(answer_counts.keys())
        
        for answer, count in answer_counts.items():
            probabilities[answer] = count / total_count
        
        logger.info(f"Generated {len(unique_options)} unique options from {total_count} samples: {answer_counts}")
        
        return probabilities, unique_options
    
    def _compute_option_probabilities(self, response: str, options: List[str]) -> Dict[str, float]:
        """Compute softmax probabilities for each option."""
        start_pos, end_pos = self._find_answer_positions(response)
        
        if start_pos == -1:
            uniform_prob = 1.0 / len(options)
            return {option: uniform_prob for option in options}
        
        probabilities = {}
        logits = []
        
        for option in options:
            modified_response = response[:start_pos] + option + response[end_pos:]
            
            inputs = self.tokenizer.encode(modified_response, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                option_tokens = self.tokenizer.encode(option, add_special_tokens=False)
                if not option_tokens:
                    logits.append(-float('inf'))
                    continue
                    
                option_token_id = option_tokens[0]
                option_pos = -1
                input_ids = inputs[0].cpu().numpy()
                
                for i in range(len(input_ids) - len(option_tokens) + 1):
                    if np.array_equal(input_ids[i:i+len(option_tokens)], option_tokens):
                        option_pos = i
                        break
                
                if option_pos > 0:
                    logit = outputs.logits[0, option_pos-1, option_token_id].item()
                else:
                    logit = outputs.logits[0, -1, option_token_id].item()
                
                logits.append(logit)
        
        logits_tensor = torch.tensor(logits)
        probs_tensor = F.softmax(logits_tensor, dim=0)
        
        for i, option in enumerate(options):
            probabilities[option] = probs_tensor[i].item()
        
        return probabilities
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample through the RAPTOR LLM system."""
        # Add documents to RAPTOR tree if context exists
        self._add_documents_to_raptor(sample)
        
        # Generate prompt
        prompt = self._generate_prompt(sample)
        
        # Extract options from the sample
        options = sample.get('options', [])
        
        if not options:
            # Case 1: Empty options - generate multiple responses and compute frequency-based probabilities
            logger.info(f"Empty options detected. Generating {self.num_samples} responses...")
            
            # Try to retrieve context from RAPTOR first
            raptor_context = self._retrieve_from_raptor(sample.get('question', ''))
            if raptor_context:
                # Use RAPTOR for answer generation
                try:
                    raptor_answer = self.raptor.answer_question(
                        question=sample.get('question', ''),
                        top_k=5,
                        max_tokens=2000
                    )
                    logger.info(f"RAPTOR generated answer: {raptor_answer}")
                    
                    # Generate multiple responses with RAPTOR context
                    enhanced_prompt = f"Context: {raptor_context}\n\n{prompt}"
                    answers = self._generate_multiple_responses(enhanced_prompt, self.num_samples)
                except Exception as e:
                    logger.warning(f"RAPTOR answer generation failed: {e}, falling back to direct generation")
                    answers = self._generate_multiple_responses(prompt, self.num_samples)
            else:
                # Fallback to direct generation
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
                'technique': sample.get('technique', 'direct'),
                'method': 'raptor_frequency_based',
                'raptor_context_used': bool(raptor_context)
            }
        else:
            # Case 2: Options provided - use RAPTOR-enhanced approach
            logger.info(f"Using provided options: {options}")
            
            # Try to retrieve context from RAPTOR
            raptor_context = self._retrieve_from_raptor(sample.get('question', ''))
            
            if raptor_context:
                # Use RAPTOR context for enhanced generation
                enhanced_prompt = f"Context: {raptor_context}\n\n{prompt}"
                response = self._generate_response(enhanced_prompt, temperature=0.1)
                logger.info("Used RAPTOR context for enhanced generation")
            else:
                # Fallback to direct generation
                response = self._generate_response(prompt, temperature=0.1)
                logger.info("Used direct generation (no RAPTOR context)")
            
            # Compute option probabilities using logits
            option_probabilities = self._compute_option_probabilities(response, options)
            
            # Extract the predicted answer
            predicted_answer = self._extract_answer(response)
            
            return {
                'id': sample.get('id', 'unknown'),
                'generated_response': response,
                'predicted_answer': predicted_answer,
                'option_probabilities': option_probabilities,
                'provided_options': options,
                'prompt_used': prompt,
                'technique': sample.get('technique', 'direct'),
                'method': 'raptor_logit_based',
                'raptor_context_used': bool(raptor_context)
            }
