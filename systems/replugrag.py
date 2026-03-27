from systems.abstract import AbstractRAGSystem
from systems.simplerag import SimpleRAGSystem
from systems.replugllm import ReplugLLMSystem, MockRetriever
from typing import Dict, Any, List
import numpy as np
from collections import defaultdict
from loguru import logger
from utils.clean import clean_web_content
from utils.ramdb import ChunkSearcher
import os
import sys

class ReplugRAGSystem(AbstractRAGSystem):
    """
    REPLUG RAG system that combines traditional RAG with REPLUG ensembling.
    
    This system implements a two-stage approach:
    1. Stage 1 (RAG): Uses traditional retrieval to get relevant documents
    2. Stage 2 (REPLUG): Ensembles multiple document contexts using REPLUG methodology
    
    The key innovation is combining:
    - RAG's semantic document retrieval
    - REPLUG's multi-document ensembling and probability weighting
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 device: str = "cuda",
                 # RAG parameters
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_retrieval_k: int = 10,
                 # REPLUG parameters  
                 ensemble_size: int = 5,
                 replug_retrieval_model: str = "facebook/contriever",
                 max_doc_length: int = 512,
                 score_weighting: bool = True,
                 # Fusion parameters
                 fusion_strategy: str = "weighted",  # 'weighted', 'max', 'average'
                 rag_weight: float = 0.6,
                 replug_weight: float = 0.4,
                 # Provided documents parameters
                 use_provided_documents: bool = False,
                 provided_documents: List[str] = None,
                 method: str = 'normal'):
        """
        Initialize ReplugRAG system.
        
        Args:
            model_name: Language model to use
            device: Device for computation
            embedding_model: Embedding model for RAG retrieval
            chunk_retrieval_k: Number of chunks to retrieve in RAG stage
            ensemble_size: Number of documents to ensemble in REPLUG stage
            replug_retrieval_model: Model for REPLUG document retrieval
            max_doc_length: Maximum document length for REPLUG
            score_weighting: Whether to weight REPLUG ensemble by retrieval scores
            fusion_strategy: How to combine RAG and REPLUG results
            rag_weight: Weight for RAG component in fusion
            replug_weight: Weight for REPLUG component in fusion
            use_provided_documents: If True, use provided_documents instead of retrieval
            provided_documents: List of documents to use directly for REPLUG (bypasses retrieval)
            method: Method to use for REPLUG
        """
        self.method = method
        replug_path = os.path.join(os.path.dirname(__file__), '..', 'REPLUG')
        if os.path.exists(replug_path) and replug_path not in sys.path:
            sys.path.insert(0, replug_path)
            
        self.rag_system = SimpleRAGSystem(
            method=method,
            model_name=model_name,
            device=device,
            embedding_model=embedding_model,
        )
        
        self.replug_system = ReplugLLMSystem(
            method=method,
            model_name=model_name,
            device=device,
            ensemble_size=ensemble_size,
            retrieval_model=replug_retrieval_model,
            max_doc_length=max_doc_length,
            score_weighting=score_weighting,
            use_provided_documents=use_provided_documents,
            provided_documents=provided_documents
        )
        
        # Configuration
        self.embedding_model = embedding_model
        self.chunk_retrieval_k = chunk_retrieval_k
        self.fusion_strategy = fusion_strategy
        self.rag_weight = rag_weight
        self.replug_weight = replug_weight
        
        # Normalize fusion weights
        total_weight = self.rag_weight + self.replug_weight
        if total_weight > 0:
            self.rag_weight /= total_weight
            self.replug_weight /= total_weight
        
        logger.info(f"ReplugRAG initialized with fusion_strategy={fusion_strategy}, "
                   f"RAG weight={self.rag_weight:.2f}, REPLUG weight={self.replug_weight:.2f}")
    
    def get_batch_size(self) -> int: return 2
    
    def set_provided_documents(self, documents: List[str], use_provided: bool = True):
        """
        Set documents to use directly for REPLUG instead of retrieval.
        
        Args:
            documents: List of document texts to use for REPLUG ensemble
            use_provided: Whether to use provided documents (True) or retrieval (False)
        """
        self.replug_system.set_provided_documents(documents, use_provided)
        logger.info(f"ReplugRAG set {len(documents)} provided documents for REPLUG component")  
    
    def _extract_rag_context(self, samples: List[Dict[str, Any]]) -> List[str]:
        """
        Extract RAG context using traditional document retrieval.
        
        Args:
            samples: List of input samples
            
        Returns:
            List of RAG contexts for each sample
        """
        try:
            # Use existing RAG system to get context
            rag_results = self.rag_system.batch_process_samples(samples)
            
            contexts = []
            for result in rag_results:
                # Extract context from RAG response or use empty string
                response = result.get('generated_response', '')
                # Try to extract any context that was used
                contexts.append(response[:500])  # Limit context length
            
            return contexts
            
        except Exception as e:
            logger.error(f"RAG context extraction failed: {e}")
            return [""] * len(samples)
    
    def _enhance_samples_with_rag(self, 
                                samples: List[Dict[str, Any]], 
                                rag_contexts: List[str]) -> List[Dict[str, Any]]:
        """
        Enhance samples with RAG-retrieved context.
        
        Args:
            samples: Original samples
            rag_contexts: RAG-retrieved contexts
            
        Returns:
            Enhanced samples with RAG context
        """
        enhanced_samples = []
        
        for sample, context in zip(samples, rag_contexts):
            enhanced_sample = sample.copy()
            
            # Add RAG context if available
            if context.strip():
                existing_context = enhanced_sample.get('context', '')
                if existing_context:
                    enhanced_sample['context'] = f"{existing_context}\n\nAdditional Context: {context}"
                else:
                    enhanced_sample['context'] = context
            
            enhanced_samples.append(enhanced_sample)
        
        return enhanced_samples
    
    def _fuse_probabilities(self, 
                           rag_probs: Dict[str, float], 
                           replug_probs: Dict[str, float]) -> Dict[str, float]:
        """
        Fuse probability distributions from RAG and REPLUG systems.
        
        Args:
            rag_probs: Probability distribution from RAG
            replug_probs: Probability distribution from REPLUG
            
        Returns:
            Fused probability distribution
        """
        # Get all possible options
        all_options = set(rag_probs.keys()) | set(replug_probs.keys())
        fused_probs = {}
        
        if self.fusion_strategy == "weighted":
            # Weighted combination
            for option in all_options:
                rag_prob = rag_probs.get(option, 0.0)
                replug_prob = replug_probs.get(option, 0.0)
                
                fused_probs[option] = (
                    self.rag_weight * rag_prob + 
                    self.replug_weight * replug_prob
                )
                
        elif self.fusion_strategy == "max":
            # Take maximum probability for each option
            for option in all_options:
                rag_prob = rag_probs.get(option, 0.0)
                replug_prob = replug_probs.get(option, 0.0)
                fused_probs[option] = max(rag_prob, replug_prob)
                
        elif self.fusion_strategy == "average":
            # Simple average
            for option in all_options:
                rag_prob = rag_probs.get(option, 0.0)
                replug_prob = replug_probs.get(option, 0.0)
                fused_probs[option] = (rag_prob + replug_prob) / 2.0
                
        else:
            logger.warning(f"Unknown fusion strategy {self.fusion_strategy}, using weighted")
            return self._fuse_probabilities(rag_probs, replug_probs)
        
        # Normalize probabilities
        total_prob = sum(fused_probs.values())
        if total_prob > 0:
            fused_probs = {k: v / total_prob for k, v in fused_probs.items()}
        else:
            # Fallback to uniform distribution
            uniform_prob = 1.0 / len(all_options) if all_options else 0.25
            fused_probs = {option: uniform_prob for option in all_options}
        
        # Convert to regular Python floats for JSON serialization
        return {k: float(v) for k, v in fused_probs.items()}
    
    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of samples using ReplugRAG methodology.
        
        Args:
            samples: List of input samples
            
        Returns:
            List of processed results with fused RAG+REPLUG information
        """
        logger.debug(f"Processing batch of {len(samples)} samples with ReplugRAG")
        
        # Stage 1: RAG Processing
        logger.debug("Stage 1: Processing with traditional RAG")
        rag_results = self.rag_system.batch_process_samples(samples)
        
        # Extract RAG contexts for enhancement
        rag_contexts = self._extract_rag_context(samples)
        enhanced_samples = self._enhance_samples_with_rag(samples, rag_contexts)
        
        # Stage 2: REPLUG Processing  
        logger.debug("Stage 2: Processing with REPLUG ensemble")
        replug_results = []
        for enhanced_sample in enhanced_samples:
            try:
                replug_result = self.replug_system.process_sample(enhanced_sample)
                replug_results.append(replug_result)
            except Exception as e:
                logger.error(f"REPLUG processing failed for sample {enhanced_sample.get('id', 'unknown')}: {e}")
                # Fallback result
                replug_results.append({
                    'id': enhanced_sample.get('id', 'unknown'),
                    'generated_response': 'REPLUG processing failed',
                    'predicted_answer': 'A',
                    'conformal_probabilities': {'A': float(0.25), 'B': float(0.25), 'C': float(0.25), 'D': float(0.25)}
                })
        
        # Stage 3: Fusion
        logger.debug("Stage 3: Fusing RAG and REPLUG results")
        fused_results = []
        
        for i, (sample, rag_result, replug_result) in enumerate(zip(samples, rag_results, replug_results)):
            try:
                # Extract probability distributions
                rag_probs = rag_result.get('conformal_probabilities', {})
                replug_probs = replug_result.get('conformal_probabilities', {})
                
                # Handle missing probabilities
                if not rag_probs:
                    predicted = rag_result.get('predicted_answer', 'A')
                    rag_probs = {predicted: 1.0}
                
                if not replug_probs:
                    predicted = replug_result.get('predicted_answer', 'A')  
                    replug_probs = {predicted: 1.0}
                
                # Fuse probabilities
                fused_probs = self._fuse_probabilities(rag_probs, replug_probs)
                
                # Get final predicted answer
                predicted_answer = max(fused_probs.items(), key=lambda x: x[1])[0]
                
                # Create fused result
                fused_result = {
                    'id': sample.get('id', f'sample_{i}'),
                    'generated_response': f"RAG: {rag_result.get('predicted_answer', 'Unknown')} | "
                                        f"REPLUG: {replug_result.get('predicted_answer', 'Unknown')}",
                    'predicted_answer': predicted_answer,
                    'conformal_probabilities': fused_probs,
                    'technique': 'replug_rag',
                    # Detailed component results
                    'rag_result': rag_result,
                    'replug_result': replug_result,
                    'fusion_strategy': self.fusion_strategy,
                    'fusion_weights': {
                        'rag': self.rag_weight,
                        'replug': self.replug_weight
                    }
                }
                
                fused_results.append(fused_result)
                
            except Exception as e:
                logger.error(f"Fusion failed for sample {sample.get('id', 'unknown')}: {e}")
                # Fallback to RAG result
                fallback_result = rag_result.copy()
                fallback_result['technique'] = 'replug_rag_fallback'
                fused_results.append(fallback_result)
        
        return fused_results
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample using ReplugRAG methodology.
        
        Args:
            sample: Input sample
            
        Returns:
            Processed result with fused RAG+REPLUG information
        """
        # Use batch processing with single sample for consistency
        results = self.batch_process_samples([sample])
        return results[0] if results else {
            'id': sample.get('id', 'unknown'),
            'generated_response': 'Processing failed',
            'predicted_answer': 'A',
            'conformal_probabilities': {'A': float(0.25), 'B': float(0.25), 'C': float(0.25), 'D': float(0.25)},
            'technique': 'replug_rag_error'
        }


class HybridChunkSearcher(ChunkSearcher):
    """
    Enhanced chunk searcher that combines semantic search with REPLUG-style retrieval.
    
    This searcher provides both traditional RAG chunks and REPLUG documents,
    allowing for more comprehensive context retrieval.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", **kwargs):
        """Initialize hybrid searcher."""
        super().__init__(embedding_model=embedding_model, **kwargs)
        
        # Ensure REPLUG is in Python path
        replug_path = os.path.join(os.path.dirname(__file__), '..', 'REPLUG')
        if os.path.exists(replug_path) and replug_path not in sys.path:
            sys.path.insert(0, replug_path)
        
        # Initialize REPLUG-style retriever for additional documents
        self.replug_retriever = None
        self.replug_enabled = True
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 10, 
                     replug_k: int = 3,
                     interaction_id: int = 0) -> tuple[List[str], List[str]]:
        """
        Perform hybrid search combining semantic chunks and REPLUG documents.
        
        Args:
            query: Search query
            k: Number of semantic chunks to retrieve
            replug_k: Number of REPLUG documents to retrieve  
            interaction_id: Interaction ID for batch processing
            
        Returns:
            Tuple of (semantic_chunks, replug_documents)
        """
        # Get traditional RAG chunks
        semantic_chunks = self.search(query, interaction_id=interaction_id, k=k)
        
        # Get REPLUG documents if available
        replug_docs = []
        if self.replug_enabled and self.replug_retriever is not None:
            try:
                docs, scores = self.replug_retriever.retrieve_passage([query])[0]
                replug_docs = [
                    doc.get('text', str(doc)) if isinstance(doc, dict) else str(doc)
                    for doc in docs[:replug_k]
                ]
            except Exception as e:
                logger.warning(f"REPLUG document retrieval failed: {e}")
        
        return semantic_chunks, replug_docs
