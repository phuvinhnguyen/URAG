import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from blingfire import text_to_sentences_and_offsets

MAX_CONTEXT_SENTENCE_LENGTH = 1000
SENTENCE_TRANSFORMER_BATCH_SIZE = 128

def process_doc(args):
    """Fast parallel document chunking"""
    doc, interaction_id, max_len = args
    if not doc.strip():
        return [], []
    
    # Fast HTML cleaning + sentence splitting
    text = BeautifulSoup(doc, "lxml").get_text(" ", strip=True) if "<" in doc else doc
    _, offsets = text_to_sentences_and_offsets(text)
    
    chunks = [text[s:e][:max_len] for s, e in offsets if e-s > 10]
    ids = [interaction_id] * len(chunks)
    return chunks, ids

class ChunkSearcher:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", max_workers: int = 8):
        self.model = SentenceTransformer(embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")
        self.max_workers = max_workers
        
    def set_documents(self, docs: List[List[str]]):
        # Parallel chunking with ThreadPool
        tasks = [(doc, i, MAX_CONTEXT_SENTENCE_LENGTH) for i, doc_list in enumerate(docs) for doc in doc_list]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_doc, tasks))
        
        # Flatten results
        chunks, ids = [], []
        for chunk_list, id_list in results:
            chunks.extend(chunk_list)
            ids.extend(id_list)
        
        self.chunks, self.ids = np.array(chunks), np.array(ids)
        
        # Single batch embedding (most efficient)
        self.embeddings = self.model.encode(chunks, normalize_embeddings=True, 
                                          batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE, show_progress_bar=False)
    
    def batch_search(self, queries: List[str], interaction_ids: List[int], k: int = 20, reverse: bool = True):
        # Single batch query embedding
        query_embeds = self.model.encode(queries, normalize_embeddings=True, 
                                       batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE, show_progress_bar=False)
        
        # Vectorized similarity matrix
        similarities = query_embeds @ self.embeddings.T
        
        def get_bot_k(args):
            i, iid = args
            mask = self.ids == (iid % (max(self.ids) + 1) if len(self.ids) > 0 else 0)
            if not mask.any():
                return []
            scores = similarities[i, mask]
            bot_idx = np.argpartition(scores, min(k, len(scores)-1))[:k]
            return self.chunks[mask][bot_idx].tolist()

        # Parallel result extraction
        def get_top_k(args):
            i, iid = args
            mask = self.ids == (iid % (max(self.ids) + 1) if len(self.ids) > 0 else 0)
            if not mask.any():
                return []
            scores = similarities[i, mask]
            top_idx = np.argpartition(-scores, min(k, len(scores)-1))[:k]
            return self.chunks[mask][top_idx].tolist()
        
        if reverse:
            with ThreadPoolExecutor(max_workers=min(len(queries), self.max_workers)) as executor:
                bot_k = list(executor.map(get_bot_k, enumerate(interaction_ids)))
                top_k = list(executor.map(get_top_k, enumerate(interaction_ids)))
                return bot_k + top_k
        else:
            with ThreadPoolExecutor(max_workers=min(len(queries), self.max_workers)) as executor:
                return list(executor.map(get_top_k, enumerate(interaction_ids)))
    
    def search(self, query: str, interaction_id: int = 0, k: int = 20):
        return self.batch_search([query], [interaction_id], k)[0]