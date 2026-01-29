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
        sample_temperature: float = 0.1,
        best_n: int = 4,
        task_description: str = "answer multi-hop questions",
        max_message_length: int = 2048,
        max_tokens_per_step: int = 128,
        corag_src_path: Optional[str] = None,
        load_corpus: bool = False,
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        vllm_api_key: str = "token-123",
        e5_dataset: Optional[str] = None,  # Dataset to filter E5 search results
        max_parallel_samples: int = 8,  # Number of samples to process in parallel
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
        self.e5_dataset = e5_dataset  # Store dataset filter
        self.max_parallel_samples = max_parallel_samples  # Parallel processing workers

        # Base LLM used for final multiple-choice answering with Answer|X format.
        # When using CoRAG with vLLM server, we don't need SimpleLLM on GPU
        # (CoRAG uses vLLM via HTTP, SimpleLLM is only fallback)
        # Use CPU to avoid GPU memory exhaustion
        fallback_device = "cpu" if device == "cuda" else device
        logger.info(f"Initializing fallback LLM on {fallback_device} (CoRAG uses vLLM server)")
        
        self.base_llm = SimpleLLMSystem(
            model_name=model_name,
            device=fallback_device,  # Use CPU for fallback
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
        
        # Thread locks for thread-safe access
        import threading
        self._init_lock = threading.Lock()  # For initialization
        self._fallback_lock = threading.Lock()  # For fallback LLM (tokenizer not thread-safe)

    def get_batch_size(self) -> int:
        # Return larger batch size since we process in parallel now
        # This allows the pipeline to feed us more samples at once
        return self.max_parallel_samples * 2
    
    def batch_process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch process samples using ThreadPoolExecutor for parallel execution.
        This significantly improves vLLM GPU utilization by processing multiple
        samples concurrently, allowing vLLM to batch requests internally.
        
        Performance: ~8x faster than sequential processing with 8 workers.
        """
        if not samples:
            return []
        
        # Use threading to process samples in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        max_workers = min(len(samples), self.max_parallel_samples)
        logger.info(f"Processing {len(samples)} samples with {max_workers} parallel workers")
        
        results = [None] * len(samples)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.process_sample, sample): idx 
                for idx, sample in enumerate(samples)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error(f"Sample {samples[idx].get('id')} generated exception: {exc}")
                    # Fallback to base LLM on error (requires lock - tokenizer not thread-safe)
                    with self._fallback_lock:
                        results[idx] = self.base_llm.process_sample(samples[idx])
        
        return results

    # ----------------------------------------------------------------------
    # Lazy init helpers
    # ----------------------------------------------------------------------
    def _load_custom_corpus(self, corpus_path: str):
        """
        Load custom corpus from JSONL file.
        
        Returns a Dataset that can be indexed like: corpus[doc_id]
        """
        import json
        from datasets import Dataset, Features, Value
        
        documents = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                # Normalize all fields to strings to avoid type inference issues
                normalized_doc = {
                    'id': str(doc.get('id', '')),
                    'title': str(doc.get('title', '')),
                    'contents': str(doc.get('contents', '')),
                    'url': str(doc.get('url', '')),
                    '_dataset': str(doc.get('_dataset', '')),
                    'source_sample_id': str(doc.get('source_sample_id', ''))
                }
                documents.append(normalized_doc)
        
        # Return as simple list instead of HuggingFace Dataset
        # Python lists are thread-safe for read-only access
        return documents
    
    def _load_combined_corpus_from_manifest(self, manifest_path: str):
        """
        Load combined corpus from all datasets in manifest.
        This matches the E5 server's global doc_id indexing.
        
        Returns a Dataset with documents from all datasets, in the order they appear in the manifest.
        """
        import json
        from datasets import Dataset, Features, Value
        
        logger.info(f"Loading combined corpus from manifest: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        all_documents = []
        for dataset_info in manifest['datasets']:
            corpus_path = dataset_info['corpus_path']
            dataset_name = dataset_info['name']
            
            logger.info(f"  Loading {dataset_name} corpus ({dataset_info['num_docs']} docs)...")
            
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    # Ensure _dataset tag is present
                    if '_dataset' not in doc:
                        doc['_dataset'] = dataset_name
                    
                    # Normalize all fields to strings to avoid type inference issues
                    # (some datasets use commit hashes as IDs, others use integers)
                    normalized_doc = {
                        'id': str(doc.get('id', '')),
                        'title': str(doc.get('title', '')),
                        'contents': str(doc.get('contents', '')),
                        'url': str(doc.get('url', '')),
                        '_dataset': str(doc.get('_dataset', dataset_name)),
                        'source_sample_id': str(doc.get('source_sample_id', ''))
                    }
                    all_documents.append(normalized_doc)
        
        logger.info(f"✅ Combined corpus loaded: {len(all_documents)} documents from {len(manifest['datasets'])} datasets")
        
        # Convert to simple list instead of HuggingFace Dataset
        # HuggingFace Dataset uses Arrow tables which are not thread-safe
        # A Python list is thread-safe for read-only access
        logger.info("Converting to thread-safe list structure...")
        return all_documents  # Return as list, not Dataset
    
    def _ensure_corag_imports(self):
        """Add corag/src to sys.path and import required modules once."""
        if self._corag_ready:
            return

        if not os.path.exists(self._corag_src_path):
            logger.warning(f"CoRAG source path not found: {self._corag_src_path}")
            self._corag_ready = False
            return

        # CRITICAL: Move corag/src to FRONT of sys.path to avoid namespace collision
        # with URAG's utils package. CoRAG has utils.py, URAG has utils/__init__.py
        if self._corag_src_path in sys.path:
            sys.path.remove(self._corag_src_path)
        sys.path.insert(0, self._corag_src_path)

        try:
            # Import using importlib to load corag's utils with explicit path
            import importlib.util
            
            # Load corag's utils.py explicitly to avoid collision
            utils_path = os.path.join(self._corag_src_path, 'utils.py')
            spec = importlib.util.spec_from_file_location("corag_utils_module", utils_path)
            corag_utils = importlib.util.module_from_spec(spec)
            sys.modules['corag_utils_module'] = corag_utils
            spec.loader.exec_module(corag_utils)
            
            # Now temporarily replace 'utils' in sys.modules so corag modules import correctly
            original_utils = sys.modules.get('utils')
            sys.modules['utils'] = corag_utils
            
            try:
                from agent import CoRagAgent  # type: ignore
                from vllm_client import VllmClient, get_vllm_model_id  # type: ignore
                from data_utils import load_corpus, format_input_context  # type: ignore
                from search.search_utils import set_search_dataset  # type: ignore
                
                # Set the E5 dataset filter if specified
                if self.e5_dataset:
                    set_search_dataset(self.e5_dataset)
                    logger.info(f"✅ E5 search will filter to dataset: {self.e5_dataset}")
                else:
                    set_search_dataset(None)
                    logger.info("E5 search will query all datasets")
                    
            finally:
                # Restore original utils module
                if original_utils is not None:
                    sys.modules['utils'] = original_utils
                else:
                    sys.modules.pop('utils', None)

            self._CoRagAgent = CoRagAgent
            self._VllmClient = VllmClient
            self._get_vllm_model_id = get_vllm_model_id
            self._load_corpus_fn = load_corpus
            self._format_input_context = format_input_context
            self._batch_truncate = corag_utils.batch_truncate
            self._corag_ready = True
            
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"CoRAG modules not available; falling back to base LLM only. Error: {exc}")
            self._corag_ready = False

    def _check_servers(self) -> bool:
        """Check if required servers are running."""
        import requests
        
        # Check vLLM
        try:
            requests.get(f"http://{self.vllm_host}:{self.vllm_port}/v1/models", timeout=2)
        except Exception:
            logger.error(f"❌ vLLM server not responding at {self.vllm_host}:{self.vllm_port}")
            logger.error(f"   Start with: cd corag && bash scripts/start_vllm_server.sh")
            return False
        
        # Check E5 search
        try:
            requests.post("http://localhost:8090/", json={"query": "test"}, timeout=2)
        except Exception:
            logger.error(f"❌ E5 search server not responding at localhost:8090")
            logger.error(f"   Start with: bash start_e5_with_custom_corpus.sh")
            return False
        
        logger.info("✅ All CoRAG servers are running")
        return True
    
    def _get_manifest_path(self) -> Optional[str]:
        """Get the path to the manifest.json file."""
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'custom_corpus')
        manifest_path = os.path.join(base_dir, 'manifest.json')
        
        if os.path.exists(manifest_path):
            return manifest_path
        return None
    
    def _ensure_agent(self):
        """Instantiate CoRAG agent, corpus, and vLLM client lazily (thread-safe)."""
        # Quick check without lock (performance optimization)
        if self._agent is not None:
            return
        
        # Acquire lock for initialization
        with self._init_lock:
            # Double-check after acquiring lock (another thread may have initialized)
            if self._agent is not None:
                return
            
            self._ensure_corag_imports()
            if not self._corag_ready:
                return
            
            # Check servers before proceeding
            if not self._check_servers():
                logger.warning("CoRAG servers not available, falling back to base LLM")
                self._corag_ready = False
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
                if self.load_corpus:
                    logger.info("Loading KILT corpus for CoRAG (this may take 1-2 minutes and ~20GB RAM)...")
                    self._corpus = self._load_corpus_fn()
                    logger.info(f"✅ KILT corpus loaded: {len(self._corpus)} passages")
                else:
                    # Try to load from manifest (multi-dataset E5 server)
                    manifest_path = self._get_manifest_path()
                    
                    if manifest_path:
                        logger.info("Detected multi-dataset setup, loading combined corpus from manifest...")
                        self._corpus = self._load_combined_corpus_from_manifest(manifest_path)
                        if self.e5_dataset:
                            logger.info(f"   E5 search will filter to dataset: {self.e5_dataset}")
                        else:
                            logger.info(f"   E5 search will query all datasets")
                    else:
                        logger.warning("⚠️  No manifest found! CoRAG needs corpus[doc_id] to retrieve documents.")
                        logger.warning("   Expected manifest at: data/custom_corpus/manifest.json")
                        logger.warning("   Options:")
                        logger.warning("   1. Generate: python generate_e5_embeddings.py")
                        logger.warning("   2. Set load_corpus=true (uses KILT Wikipedia)")
                        logger.warning("   3. Use simplerag instead")
                        logger.warning("   Disabling CoRAG, will use fallback LLM only.")
                        self._corpus = None
                        self._corag_ready = False
                        return
            except Exception as exc:
                logger.warning(f"Could not load corpus; disabling CoRAG. Error: {exc}")
                self._corpus = None
                self._corag_ready = False
                return

            try:
                self._agent = self._CoRagAgent(vllm_client=self._vllm_client, corpus=self._corpus)
                logger.info(f"✅ CoRAG agent initialized successfully (corpus shared across {self.max_parallel_samples} workers)")
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
                # Corpus is now a simple Python list (thread-safe for reads)
                for doc_id in unique_doc_ids:
                    try:
                        documents.append(self._format_input_context(self._corpus[int(doc_id)]))
                    except Exception:
                        continue

            # Truncate overly long documents to keep prompts manageable.
            if documents and self._batch_truncate:
                # Tokenizer access requires lock (not thread-safe)
                with self._fallback_lock:
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

    @staticmethod
    def _try_extract_answer_letter(text: str, options: List[str]) -> Optional[str]:
        """Return a matching option letter if found; otherwise None."""
        if not text or not options:
            return None

        match = re.search(r"Answer\|([A-Z])", text)
        if match and match.group(1) in options:
            return match.group(1)

        for opt in options:
            if len(opt) == 1 and opt.isalpha():
                if re.search(rf"\b{re.escape(opt)}\b", text, flags=re.IGNORECASE):
                    return opt

        return None

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

        # Prefer vLLM final answer when possible; fallback to base LLM if parsing fails.
        vllm_pred = self._try_extract_answer_letter(final_answer or "", options)
        if vllm_pred:
            generated_response = final_answer or ""
            if "Answer|" not in generated_response:
                generated_response = f"{generated_response}\nAnswer|{vllm_pred}".strip()

            conformal_probs = {opt: 0.0 for opt in options} if options else {vllm_pred: 1.0}
            if options:
                conformal_probs[vllm_pred] = 1.0

            return {
                "id": sample.get("id", "unknown"),
                "generated_response": generated_response,
                "predicted_answer": vllm_pred,
                "conformal_probabilities": conformal_probs,
                "technique": "corag_llm",
                "corag_path": {
                    "subqueries": getattr(path, "past_subqueries", None) if path else None,
                    "subanswers": getattr(path, "past_subanswers", None) if path else None,
                    "doc_ids": getattr(path, "past_doc_ids", None) if path else None,
                },
                "corag_documents": documents,
                "corag_final_answer": final_answer,
                "e5_dataset": self.e5_dataset,
            }

        # Build augmented context and delegate final MC answer to base LLM.
        context = self._build_context_block(documents, path, final_answer)
        llm_sample = sample.copy()
        if context:
            llm_sample["context"] = context

        # Base LLM uses tokenizer - requires lock for thread safety
        with self._fallback_lock:
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
                "e5_dataset": self.e5_dataset,
            }
        )

        return base_result
