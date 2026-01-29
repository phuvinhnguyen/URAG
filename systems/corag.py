from systems.abstract import AbstractRAGSystem
from systems.simplerag import SimpleRAGSystem
from systems.collm import CoRAGLLMSystem
from typing import Dict, Any, List, Optional
from loguru import logger


class CoRAGSystem(AbstractRAGSystem):
    """
    RAG wrapper that blends traditional retrieval with the CoRAG multi-hop agent.

    The pipeline runs a baseline SimpleRAG pass to obtain quick semantic context,
    then invokes the CoRAGLLMSystem to gather multi-hop evidence and answer the
    question. Probabilities from both components are fused for robustness.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        fusion_strategy: str = "weighted",
        rag_weight: float = 0.5,
        corag_weight: float = 0.5,
        method: str = "normal",
        e5_dataset: Optional[str] = None,  # Dataset to filter E5 search results
        max_parallel_samples: int = 8,  # Number of samples to process in parallel
        **kwargs,
    ):
        self.method = method
        self.fusion_strategy = fusion_strategy
        self.rag_weight = rag_weight
        self.corag_weight = corag_weight
        self.e5_dataset = e5_dataset

        # Normalize weights to avoid scaling issues.
        total = self.rag_weight + self.corag_weight
        if total > 0:
            self.rag_weight /= total
            self.corag_weight /= total
        else:
            # Default to CoRAG-only if both weights are zero/invalid.
            logger.warning("Both rag_weight and corag_weight are 0; defaulting to CoRAG-only.")
            self.rag_weight = 0.0
            self.corag_weight = 1.0

        self.rag_system = None
        self.corag_system = None

        if self.rag_weight > 0:
            self.rag_system = SimpleRAGSystem(model_name=model_name, device=device, method=method)
        else:
            logger.info("RAG weight is 0; skipping SimpleRAG initialization.")

        if self.corag_weight > 0:
            self.corag_system = CoRAGLLMSystem(
                model_name=model_name,
                device=device,
                method=method,
                e5_dataset=e5_dataset,  # Pass dataset filter to CoRAG
                max_parallel_samples=max_parallel_samples,  # Enable parallel processing
                **kwargs,
            )
        else:
            logger.info("CoRAG weight is 0; skipping CoRAG initialization.")

    def get_batch_size(self) -> int:
        # Limited by the slower CoRAG branch.
        if self.rag_system and self.corag_system:
            return min(self.rag_system.get_batch_size(), self.corag_system.get_batch_size())
        if self.rag_system:
            return self.rag_system.get_batch_size()
        if self.corag_system:
            return self.corag_system.get_batch_size()
        raise RuntimeError("Both RAG and CoRAG systems are disabled; cannot determine batch size.")

    def _fuse_probabilities(self, rag_probs: Dict[str, float], corag_probs: Dict[str, float]) -> Dict[str, float]:
        all_options = set(rag_probs.keys()) | set(corag_probs.keys())
        fused: Dict[str, float] = {}

        if self.fusion_strategy == "max":
            for opt in all_options:
                fused[opt] = max(rag_probs.get(opt, 0.0), corag_probs.get(opt, 0.0))
        elif self.fusion_strategy == "average":
            for opt in all_options:
                fused[opt] = (rag_probs.get(opt, 0.0) + corag_probs.get(opt, 0.0)) / 2.0
        else:  # default weighted
            for opt in all_options:
                fused[opt] = self.rag_weight * rag_probs.get(opt, 0.0) + self.corag_weight * corag_probs.get(opt, 0.0)

        total = sum(fused.values())
        if total > 0:
            fused = {k: float(v / total) for k, v in fused.items()}
        else:
            uniform = 1.0 / len(all_options) if all_options else 1.0
            fused = {opt: uniform for opt in all_options}

        return fused

    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.rag_system is None and self.corag_system is None:
            raise RuntimeError("Both RAG and CoRAG systems are disabled; cannot process samples.")

        if self.rag_system is None:
            return self.corag_system.batch_process_samples(samples)
        if self.corag_system is None:
            return self.rag_system.batch_process_samples(samples)

        rag_results = self.rag_system.batch_process_samples(samples)
        # Use CoRAG's batch processing for parallel execution (better vLLM utilization)
        corag_results = self.corag_system.batch_process_samples(samples)

        fused_results: List[Dict[str, Any]] = []
        for sample, rag_result, corag_result in zip(samples, rag_results, corag_results):
            try:
                rag_probs = rag_result.get("conformal_probabilities", {}) or {}
                corag_probs = corag_result.get("conformal_probabilities", {}) or {}

                if not rag_probs:
                    pred = rag_result.get("predicted_answer", "A")
                    rag_probs = {pred: 1.0}
                if not corag_probs:
                    pred = corag_result.get("predicted_answer", "A")
                    corag_probs = {pred: 1.0}

                fused_probs = self._fuse_probabilities(rag_probs, corag_probs)
                predicted_answer = max(fused_probs.items(), key=lambda x: x[1])[0]

                fused_results.append(
                    {
                        "id": sample.get("id", "unknown"),
                        "generated_response": f"RAG: {rag_result.get('predicted_answer', 'Unknown')} | CoRAG: {corag_result.get('predicted_answer', 'Unknown')}",
                        "predicted_answer": predicted_answer,
                        "conformal_probabilities": fused_probs,
                        "technique": "corag_rag",
                        "rag_result": rag_result,
                        "corag_result": corag_result,
                        "fusion_strategy": self.fusion_strategy,
                        "fusion_weights": {"rag": self.rag_weight, "corag": self.corag_weight},
                        "e5_dataset": self.e5_dataset,
                    }
                )
            except Exception as exc:
                logger.error(f"Fusion failed for sample {sample.get('id', 'unknown')}: {exc}")
                fallback = rag_result.copy()
                fallback["technique"] = "corag_rag_fallback"
                fused_results.append(fallback)

        return fused_results

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Reuse batch logic for single sample to keep consistent behavior.
        return self.batch_process_samples([sample])[0]
