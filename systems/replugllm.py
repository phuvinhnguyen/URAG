from systems.abstract import AbstractRAGSystem
from systems.simplellm import SimpleLLMSystem
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
from collections import defaultdict
from loguru import logger

class ReplugLLMSystem(AbstractRAGSystem):
    """
    REPLUG LLM system that uses document ensembling for improved performance.
    
    This system follows the REPLUG methodology:
    1. Retrieves multiple documents for each query
    2. Generates responses using each document as context
    3. Ensembles the results using weighted probabilities
    4. Returns the best answer based on ensemble scoring
    
    Unlike SimpleRAG, this focuses on LLM-based ensembling rather than traditional RAG.
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 device: str = "cuda",
                 technique: str = "replug",
                 max_new_tokens: int = 100,
                 temperature: float = 0.1,
                 ensemble_size: int = 5,
                 retrieval_model: str = "facebook/contriever",
                 max_doc_length: int = 1024,
                 score_weighting: bool = True,
                 use_provided_documents: bool = False,
                 provided_documents: List[str] = None,
                 method: str = 'normal'):
        """
        Initialize ReplugLLM system.
        
        Args:
            model_name: Language model to use
            device: Device for computation
            technique: Technique identifier 
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            ensemble_size: Number of documents to ensemble
            retrieval_model: Model for document retrieval
            max_doc_length: Maximum document length
            score_weighting: Whether to weight ensemble by retrieval scores
            use_provided_documents: If True, use provided_documents instead of retrieval
            provided_documents: List of documents to use directly (bypasses retrieval)
            method: Method to use for REPLUG
        """
        # Initialize base LLM system
        self.method = method
        self.base_llm = SimpleLLMSystem(
            model_name=model_name,
            device=device,
            technique='rag',  # Use RAG technique for context handling
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        self.ensemble_size = ensemble_size
        self.retrieval_model = retrieval_model
        self.max_doc_length = max_doc_length
        self.score_weighting = score_weighting
        self.technique = technique
        self.retriever = None
        self.use_provided_documents = use_provided_documents
        self.provided_documents = provided_documents or []
        
        logger.info(f"ReplugLLM initialized with ensemble_size={ensemble_size}")
    
    def get_batch_size(self) -> int: return 20
    
    def set_provided_documents(self, documents: List[str], use_provided: bool = True):
        """
        Set documents to use directly instead of retrieval.
        
        Args:
            documents: List of document texts to use for ensemble
            use_provided: Whether to use provided documents (True) or retrieval (False)
        """
        self.provided_documents = documents
        self.use_provided_documents = use_provided
        logger.info(f"Set {len(documents)} provided documents, use_provided={use_provided}")
    
    def _init_retriever(self):
        """Initialize REPLUG retriever with fallback to mock retriever."""
        if self.retriever is not None:
            return
            
        try:
            # Add REPLUG to Python path
            import os
            import sys
            import torch
            replug_path = os.path.join(os.path.dirname(__file__), '..', 'REPLUG')
            if os.path.exists(replug_path) and replug_path not in sys.path:
                sys.path.insert(0, replug_path)
            
            # Try to use actual REPLUG retriever
            from REPLUG.retriever import Retriever
            from argparse import Namespace
            
            # Create comprehensive args with all required parameters
            # Use absolute paths for robustness
            base_dir = os.path.dirname(__file__)
            replug_data_path = os.path.abspath(os.path.join(base_dir, '..', 'REPLUG', 'data'))
            
            args = Namespace(
                re_model_name_or_path=self.retrieval_model,
                passages_embeddings=os.path.join(replug_data_path, "embeddings", "passages_*"),
                passages=os.path.join(replug_data_path, "text.jsonl"),
                n_docs=self.ensemble_size,
                projection_size=768,
                n_subquantizers=0,
                n_bits=8,
                # Additional required parameters
                save_or_load_index=True,
                indexing_batch_size=1000,
                use_faiss_gpu=torch.cuda.is_available() and "cuda" in self.base_llm.device,
                num_gpus=1 if torch.cuda.is_available() and "cuda" in self.base_llm.device else 0,
                cache_dict=os.path.join(replug_data_path, "..", "cache_dict.pkl"),
                chunk_size=100,
                per_gpu_batch_size=16,
                question_maxlength=512,
                normalize_text=True,
                ra_truncate_broken_sents=False,
                ra_round_broken_sents=False
            )
            
            self.retriever = Retriever(args)
            logger.info("Successfully initialized REPLUG retriever")
            
        except Exception as e:
            logger.warning(f"Could not initialize REPLUG retriever: {e}")
            logger.info("Using mock retriever for testing")
            self.retriever = MockRetriever(self.ensemble_size)
    
    def _retrieve_documents(self, query: str) -> tuple[List[str], List[float]]:
        """
        Retrieve documents for a given query or use provided documents.
        
        Args:
            query: Query string (ignored if using provided documents)
            
        Returns:
            Tuple of (documents, scores)
        """
        # If using provided documents, return them directly
        if self.use_provided_documents and self.provided_documents:
            logger.debug(f"Using {len(self.provided_documents)} provided documents")
            
            # Process provided documents
            doc_texts = []
            for doc in self.provided_documents[:self.ensemble_size]:
                if isinstance(doc, dict):
                    text = doc.get('text', str(doc))
                else:
                    text = str(doc)
                
                # Truncate document if too long
                if len(text) > self.max_doc_length:
                    text = text[:self.max_doc_length] + "..."
                doc_texts.append(text)
            
            # Create uniform scores for provided documents
            scores = [float(1.0)] * len(doc_texts)
            
            # Pad with empty documents if needed
            while len(doc_texts) < self.ensemble_size:
                doc_texts.append("")
                scores.append(float(1.0))
                
            return doc_texts[:self.ensemble_size], scores[:self.ensemble_size]
        
        # Otherwise, use standard retrieval
        self._init_retriever()
        
        try:
            results = self.retriever.retrieve_passage([query])
            if results and len(results) > 0:
                docs, scores = results[0]
                # Extract text from document objects
                doc_texts = []
                for doc in docs[:self.ensemble_size]:
                    if isinstance(doc, dict):
                        text = doc.get('text', str(doc))
                    else:
                        text = str(doc)
                    
                    # Truncate document if too long
                    if len(text) > self.max_doc_length:
                        text = text[:self.max_doc_length] + "..."
                    doc_texts.append(text)
                
                # Convert scores to regular Python floats
                return doc_texts, [float(score) for score in scores[:self.ensemble_size]]
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
        
        # Fallback to empty documents
        return [""] * self.ensemble_size, [float(1.0)] * self.ensemble_size
    
    def _generate_ensemble_responses(self, 
                                   sample: Dict[str, Any], 
                                   documents: List[str], 
                                   scores: List[float]) -> List[Dict[str, Any]]:
        """
        Generate responses using each document as context.
        
        Args:
            sample: Input sample
            documents: Retrieved documents
            scores: Retrieval scores
            
        Returns:
            List of response dictionaries
        """
        ensemble_responses = []
        
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # Create sample with document context
            doc_sample = sample.copy()
            if doc.strip():  # Only add context if document is not empty
                doc_sample['context'] = doc
            
            try:
                # Generate response using base LLM
                response = self.base_llm.process_sample(doc_sample)
                
                # Add ensemble metadata
                response['ensemble_id'] = i
                response['retrieval_score'] = float(score)
                response['document'] = doc
                
                ensemble_responses.append(response)
                
            except Exception as e:
                logger.error(f"Error generating response for ensemble {i}: {e}")
                # Add fallback response
                ensemble_responses.append({
                    'ensemble_id': i,
                    'generated_response': 'Error in generation',
                    'predicted_answer': 'A',
                    'conformal_probabilities': {'A': float(0.25), 'B': float(0.25), 'C': float(0.25), 'D': float(0.25)},
                    'retrieval_score': float(score),
                    'document': doc
                })
        
        return ensemble_responses
    
    def _ensemble_probabilities(self, 
                              ensemble_responses: List[Dict[str, Any]],
                              use_retrieval_scores: bool = True) -> Dict[str, float]:
        """
        Ensemble probabilities from multiple responses.
        
        Args:
            ensemble_responses: List of response dictionaries
            use_retrieval_scores: Whether to weight by retrieval scores
            
        Returns:
            Combined probability distribution
        """
        if not ensemble_responses:
            return {'A': float(0.25), 'B': float(0.25), 'C': float(0.25), 'D': float(0.25)}
        
        # Get all possible options
        all_options = set()
        for response in ensemble_responses:
            probs = response.get('conformal_probabilities', {})
            all_options.update(probs.keys())
        
        if not all_options:
            all_options = {'A', 'B', 'C', 'D'}
        
        # Initialize ensemble probabilities
        ensemble_probs = defaultdict(float)
        total_weight = 0.0
        
        for response in ensemble_responses:
            probs = response.get('conformal_probabilities', {})
            retrieval_score = response.get('retrieval_score', 1.0)
            
            # Use retrieval score as weight if enabled
            weight = retrieval_score if use_retrieval_scores and self.score_weighting else 1.0
            weight = max(weight, 1e-10)  # Avoid zero weights
            
            for option in all_options:
                prob = probs.get(option, 1.0 / len(all_options))  # Uniform fallback
                ensemble_probs[option] += prob * weight
                
            total_weight += weight
        
        # Normalize probabilities
        if total_weight > 0:
            for option in ensemble_probs:
                ensemble_probs[option] /= total_weight
        else:
            # Uniform distribution fallback
            uniform_prob = 1.0 / len(all_options)
            for option in all_options:
                ensemble_probs[option] = uniform_prob
        
        # Convert to regular Python floats for JSON serialization
        return {k: float(v) for k, v in ensemble_probs.items()}
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample using REPLUG methodology.
        
        Args:
            sample: Input sample with question and options
            
        Returns:
            Processed result with ensemble information
        """
        question = sample.get('question', '')
        sample_id = sample.get('id', 'unknown')
        
        logger.debug(f"Processing sample {sample_id} with REPLUG")
        
        # Retrieve documents
        documents, scores = self._retrieve_documents(question)
        
        # Generate ensemble responses
        ensemble_responses = self._generate_ensemble_responses(sample, documents, scores)
        
        # Ensemble probabilities
        ensemble_probs = self._ensemble_probabilities(
            ensemble_responses, 
            use_retrieval_scores=self.score_weighting
        )
        
        # Get predicted answer
        predicted_answer = max(ensemble_probs.items(), key=lambda x: x[1])[0]
        
        # Create combined response
        combined_response = " | ".join([
            resp.get('predicted_answer', 'Unknown') 
            for resp in ensemble_responses
        ])
        
        return {
            'id': sample_id,
            'generated_response': combined_response,
            'predicted_answer': predicted_answer,
            'conformal_probabilities': ensemble_probs,
            'technique': self.technique,
            # REPLUG-specific data
            'ensemble_responses': ensemble_responses,
            'retrieved_documents': documents,
            'retrieval_scores': scores,
            'ensemble_size': len(ensemble_responses)
        }


class MockRetriever:
    """Mock retriever for testing when REPLUG components are not available."""
    
    def __init__(self, n_docs: int = 5):
        self.n_docs = n_docs
        
        self.mock_docs = [
            {"text": "This is a sample document about science and mathematics."},
            {"text": "Historical information and context for various topics."},
            {"text": "General knowledge and factual information database."},
            {"text": "Educational content covering multiple subject areas."},
            {"text": "Reference material for questions and answers."}
        ]
    
    def retrieve_passage(self, queries: List[str]) -> List[tuple]:
        """Mock retrieval that returns sample documents."""
        results = []
        for query in queries:
            docs = self.mock_docs[:self.n_docs]
            scores = [float(1.0 - (i * 0.1)) for i in range(len(docs))]
            results.append((docs, scores))
        
        return results
