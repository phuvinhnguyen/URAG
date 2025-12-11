from systems.abstract import AbstractRAGSystem
from systems.simplellm import SimpleLLMSystem
from loguru import logger
from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import re


class CoRAGLLMSystem(AbstractRAGSystem):
    """
    Chain-of-Retrieval Augmented Generation (CoRAG) system.

    This integrates the upstream CoRAG implementation (corag/src) and wraps its
    multi-hop retrieval traces into the existing multiple-choice evaluation
    pipeline. The CoRAG agent is used to propose sub-queries and collect
    supporting documents; the final multiple-choice decision is delegated to
    the baseline SimpleLLMSystem so the outputs still follow the Answer|X
    convention expected by the evaluation pipeline.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        method: str = "normal",
        max_path_length: int = 3,
        decode_strategy: str = "greedy",
        sample_temperature: float = 0.7,
        best_n: int = 4,
        task_description: str = "answer multi-hop questions",
        max_message_length: int = 4096,
        max_tokens_per_step: int = 128,
        corag_src_path: Optional[str] = None,
        load_corpus: bool = False,
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        vllm_api_key: str = "token-123",
    ):
        self.method = method
        self.max_path_length = max_path_length
        self.decode_strategy = decode_strategy
        self.sample_temperature = sample_temperature
        self.best_n = best_n
        self.task_description = task_description
        self.max_message_length = max_message_length
        self.max_tokens_per_step = max_tokens_per_step
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.vllm_api_key = vllm_api_key
        self.load_corpus = load_corpus

        # Base LLM used for final multiple-choice answering with Answer|X format.
        self.base_llm = SimpleLLMSystem(
            model_name=model_name,
            device=device,
            technique="rag",
            max_new_tokens=max_tokens_per_step,
            temperature=sample_temperature,
            method=method,
        )

        # CoRAG internals are loaded lazily so the system still works when the
        # optional CoRAG dependencies (vLLM server, HF datasets) are unavailable.
        self._corag_ready = False
        self._corag_src_path = corag_src_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "corag", "src")
        )
        self._agent = None
        self._corpus = None
        self._vllm_client = None

    def get_batch_size(self) -> int:
        # CoRAG internally runs multiple network calls; keep batch size modest.
        return 4

    # ----------------------------------------------------------------------
    # Lazy init helpers
    # ----------------------------------------------------------------------
    def _ensure_corag_imports(self):
        """Add corag/src to sys.path and import required modules once."""
        if self._corag_ready:
            return

        if os.path.exists(self._corag_src_path) and self._corag_src_path not in sys.path:
            sys.path.insert(0, self._corag_src_path)

        try:
            from agent import CoRagAgent  # type: ignore
            from vllm_client import VllmClient, get_vllm_model_id  # type: ignore
            from data_utils import load_corpus, format_input_context  # type: ignore
            from utils import batch_truncate  # type: ignore

            self._CoRagAgent = CoRagAgent
            self._VllmClient = VllmClient
            self._get_vllm_model_id = get_vllm_model_id
            self._load_corpus_fn = load_corpus
            self._format_input_context = format_input_context
            self._batch_truncate = batch_truncate
            self._corag_ready = True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"CoRAG modules not available; falling back to base LLM only. Error: {exc}")
            self._corag_ready = False

    def _ensure_agent(self):
        """Instantiate CoRAG agent, corpus, and vLLM client lazily."""
        self._ensure_corag_imports()
        if not self._corag_ready or self._agent is not None:
            return

        try:
            model_id = self._get_vllm_model_id(self.vllm_host, self.vllm_port, self.vllm_api_key)
        except Exception as exc:
            # If listing models fails, fall back to provided model_name string.
            logger.warning(f"Could not fetch vLLM model id, using base model name. Error: {exc}")
            model_id = self.base_llm.tokenizer.name_or_path

        try:
            self._vllm_client = self._VllmClient(model=model_id, host=self.vllm_host, port=self.vllm_port, api_key=self.vllm_api_key)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to create vLLM client, CoRAG will be disabled. Error: {exc}")
            self._corag_ready = False
            return

        try:
            self._corpus = self._load_corpus_fn() if self.load_corpus else None
        except Exception as exc:
            logger.warning(f"Could not load CoRAG corpus; proceeding without documents. Error: {exc}")
            self._corpus = None

        try:
            self._agent = self._CoRagAgent(vllm_client=self._vllm_client, corpus=self._corpus)
        except Exception as exc:
            logger.warning(f"Failed to initialize CoRAG agent, disabling CoRAG. Error: {exc}")
            self._corag_ready = False

    # ----------------------------------------------------------------------
    # Processing
    # ----------------------------------------------------------------------
    def _run_corag_path(self, question: str) -> Tuple[Optional[Any], List[str]]:
        """
        Run the CoRAG agent to sample a reasoning path and collect document texts.
        Returns (path, documents) where either may be empty on failure.
        """
        self._ensure_agent()
        if not self._corag_ready or self._agent is None:
            return None, []

        path = None
        try:
            if self.decode_strategy == "tree_search":
                path = self._agent.tree_search(
                    query=question,
                    task_desc=self.task_description,
                    max_path_length=self.max_path_length,
                    max_message_length=self.max_message_length,
                    temperature=self.sample_temperature,
                )
            elif self.decode_strategy == "best_of_n":
                path = self._agent.best_of_n(
                    query=question,
                    task_desc=self.task_description,
                    max_path_length=self.max_path_length,
                    max_message_length=self.max_message_length,
                    temperature=self.sample_temperature,
                    n=self.best_n,
                )
            else:
                path = self._agent.sample_path(
                    query=question,
                    task_desc=self.task_description,
                    max_path_length=self.max_path_length,
                    max_message_length=self.max_message_length,
                    temperature=self.sample_temperature,
                    max_tokens=self.max_tokens_per_step,
                )
        except Exception as exc:
            logger.warning(f"CoRAG path generation failed; fallback to base LLM. Error: {exc}")
            return None, []

        documents: List[str] = []
        try:
            doc_ids = []
            for step_doc_ids in path.past_doc_ids or []:
                doc_ids.extend(step_doc_ids)

            seen = set()
            unique_doc_ids = []
            for doc_id in doc_ids:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                unique_doc_ids.append(doc_id)

            if self._corpus is not None and self._format_input_context:
                for doc_id in unique_doc_ids:
                    try:
                        documents.append(self._format_input_context(self._corpus[int(doc_id)]))
                    except Exception:
                        continue

            # Truncate overly long documents to keep prompts manageable.
            if documents and self._batch_truncate:
                documents = self._batch_truncate(
                    documents,
                    tokenizer=self.base_llm.tokenizer,
                    max_length=min(1024, self.max_message_length),
                    truncate_from_middle=True,
                    skip_special_tokens=True,
                )
        except Exception as exc:
            logger.debug(f"Failed to format CoRAG documents: {exc}")
            documents = []

        return path, documents

    @staticmethod
    def _extract_answer_letter(text: str, options: List[str]) -> str:
        """Heuristically extract the option letter from the generated text."""
        if not options:
            return "A"

        # Look for Answer|X style first.
        match = re.search(r"Answer\|([A-Z])", text)
        if match and match.group(1) in options:
            return match.group(1)

        # Otherwise try to match single-letter option tokens.
        for opt in options:
            if len(opt) == 1 and opt.isalpha():
                if re.search(rf"\b{re.escape(opt)}\b", text, flags=re.IGNORECASE):
                    return opt

        # Fallback to first option.
        return options[0]

    def _build_context_block(self, documents: List[str], path: Optional[Any], final_answer: Optional[str]) -> str:
        sections = []
        if documents:
            sections.append("Retrieved documents:\n- " + "\n- ".join(documents[:5]))
        if path is not None and getattr(path, "past_subqueries", None):
            steps = []
            for idx, (q, a) in enumerate(zip(path.past_subqueries or [], path.past_subanswers or [])):
                steps.append(f"Step {idx + 1} Q: {q}\nA: {a}")
            sections.append("CoRAG intermediate steps:\n" + "\n".join(steps))
        if final_answer:
            sections.append(f"CoRAG final answer suggestion: {final_answer}")
        return "\n\n".join(sections)[:4000]

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run CoRAG to gather evidence, then answer the multiple-choice question
        with the base LLM using the collected context.
        """
        question = sample.get("question", "")
        options = sample.get("options", ["A", "B", "C", "D"])

        path, documents = self._run_corag_path(question)
        final_answer = None

        if self._corag_ready and self._agent is not None and path is not None:
            try:
                final_answer = self._agent.generate_final_answer(
                    corag_sample=path,
                    task_desc=self.task_description,
                    documents=documents if documents else None,
                    max_message_length=self.max_message_length,
                    temperature=0.0,
                    max_tokens=self.max_tokens_per_step,
                )
            except Exception as exc:
                logger.debug(f"CoRAG final answer generation failed: {exc}")

        # Build augmented context and delegate final MC answer to base LLM.
        context = self._build_context_block(documents, path, final_answer)
        llm_sample = sample.copy()
        if context:
            llm_sample["context"] = context

        base_result = self.base_llm.process_sample(llm_sample)

        # Ensure a predicted answer is present even if base LLM fails.
        predicted_answer = base_result.get("predicted_answer")
        if not predicted_answer:
            predicted_answer = self._extract_answer_letter(
                base_result.get("generated_response", ""), options
            )
            base_result["predicted_answer"] = predicted_answer

        # Defensive conformal probabilities.
        conformal_probs = base_result.get("conformal_probabilities")
        if not conformal_probs or not isinstance(conformal_probs, dict):
            uniform = 1.0 / len(options) if options else 1.0
            conformal_probs = {opt: uniform for opt in options}
            base_result["conformal_probabilities"] = conformal_probs

        base_result.update(
            {
                "technique": "corag_llm",
                "corag_path": {
                    "subqueries": getattr(path, "past_subqueries", None) if path else None,
                    "subanswers": getattr(path, "past_subanswers", None) if path else None,
                    "doc_ids": getattr(path, "past_doc_ids", None) if path else None,
                },
                "corag_documents": documents,
                "corag_final_answer": final_answer,
            }
        )

        return base_result

