
from collections import defaultdict
from typing import List
import numpy as np
import ray
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import torch

#### CONFIG PARAMETERS ---
NUM_CONTEXT_SENTENCES = 20
MAX_CONTEXT_SENTENCE_LENGTH = 1000
MAX_CONTEXT_REFERENCES_LENGTH = 4000
SENTENCE_TRANSFORMER_BATCH_SIZE = 128
#### CONFIG PARAMETERS END ---

class ChunkExtractor:
    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)

        if not text: return interaction_id, [""]

        _, offsets = text_to_sentences_and_offsets(text)

        chunks = []

        for start, end in offsets:
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_page_results):
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text
            )
            for idx, page_results in enumerate(batch_page_results)
            for html_text in page_results
        ]

        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

class ChunkSearcher:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", max_sentence_length: int = MAX_CONTEXT_SENTENCE_LENGTH, batch_size: int = SENTENCE_TRANSFORMER_BATCH_SIZE):
        self.embedding_model_name = embedding_model
        self.max_sentence_length = max_sentence_length
        self.batch_size = batch_size
        self.sentence_model = SentenceTransformer(
            self.embedding_model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.chunk_extractor = ChunkExtractor()

    def calculate_embeddings(self, sentences):
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )
        return embeddings
    
    def set_documents(self, documents: List[List[str]]):
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks([i for i in range(len(documents))], documents)
        self.chunk_embeddings = self.calculate_embeddings(chunks)
        self.chunks = chunks
        self.chunk_interaction_ids = chunk_interaction_ids
        
    def search(self, query: str, interaction_id = 0, k: int = NUM_CONTEXT_SENTENCES):
        query_embedding = self.calculate_embeddings([query])[0]
        relevant_chunks_mask = self.chunk_interaction_ids == interaction_id
        relevant_chunks = self.chunks[relevant_chunks_mask]
        relevant_chunks_embeddings = self.chunk_embeddings[relevant_chunks_mask]
        cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)
        retrieval_results = relevant_chunks[
            (-cosine_scores).argsort()[:k]
        ]
        return retrieval_results
