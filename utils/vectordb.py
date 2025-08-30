import os
import json
import pickle
import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import uuid
import tempfile
import shutil
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    print("Warning: qdrant-client not installed. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False


@dataclass
class ChunkConfig:
    """Configuration for text chunking"""
    chunk_size: int = 500
    overlap: int = 50
    chunking_method: str = "sliding_window"  # "sliding_window", "sentence", "paragraph"


@dataclass
class DatabaseConfig:
    """Configuration for a single database"""
    name: str
    embedding_model: str = "tfidf"  # "tfidf", "sentence_transformers", "openai"
    chunk_config: ChunkConfig = None
    
    def __post_init__(self):
        if self.chunk_config is None:
            self.chunk_config = ChunkConfig()


class BM25:
    """Lightweight BM25 implementation with reduced memory usage"""
    
    def __init__(self, corpus: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)
        self.doc_len = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.doc_count if self.doc_count > 0 else 0
        
        # More memory-efficient word frequency calculation
        word_doc_count = defaultdict(int)
        self.doc_freqs = []
        
        for doc in corpus:
            frequencies = defaultdict(int)
            doc_words = set()
            for word in doc.split():
                frequencies[word] += 1
                if word not in doc_words:
                    word_doc_count[word] += 1
                    doc_words.add(word)
            self.doc_freqs.append(frequencies)
        
        # Calculate IDF values only for words that appear
        self.idf = {}
        for word, doc_count in word_doc_count.items():
            self.idf[word] = np.log((self.doc_count - doc_count + 0.5) / (doc_count + 0.5))
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search using BM25 scoring"""
        query_words = query.split()
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            for word in query_words:
                if word in doc_freq:
                    freq = doc_freq[word]
                    idf = self.idf.get(word, 0)
                    score += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class QdrantVectorDB:
    """
    Vector Database using Qdrant for efficient storage and retrieval
    Significantly reduces RAM usage and improves search performance
    """
    
    def __init__(self, 
                 texts: Union[List[str], List[List[str]]], 
                 embedding_model: str = "tfidf",
                 chunk_size: int = 500,
                 overlap: int = 50,
                 chunking_method: str = "sliding_window",
                 db_names: Optional[List[str]] = None,
                 qdrant_path: Optional[str] = None):
        """
        Initialize the vector database with Qdrant backend
        
        Args:
            texts: Single list of texts (1 DB) or list of lists (multiple DBs)
            embedding_model: Type of embedding ("tfidf", "sentence_transformers", "openai")
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            chunking_method: Method for chunking ("sliding_window", "sentence", "paragraph")
            db_names: Names for databases (auto-generated if None)
            qdrant_path: Path for Qdrant storage (temp dir if None)
        """
        
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required. Install with: pip install qdrant-client")
        
        # Store the user-provided embedding model string. This can be a type or a concrete model name
        self.embedding_model = embedding_model
        # Determine embedding model type and, if applicable, the sentence-transformers model name
        known_types = {"tfidf", "openai", "sentence_transformers"}
        if embedding_model in known_types:
            self.embedding_model_type = embedding_model
            # Default model name for sentence-transformers when type is specified
            self.sentence_transformer_model_name = 'all-MiniLM-L6-v2' if embedding_model == "sentence_transformers" else None
        else:
            # If a specific model name is provided (e.g., 'all-MiniLM-L6-v2'), use sentence-transformers automatically
            self.embedding_model_type = "sentence_transformers"
            self.sentence_transformer_model_name = embedding_model
        self.current_db_index = 0
        self.db_names = []
        self.collections = {}  # Maps db_index to collection_name
        self.bm25_indices = {}  # Store BM25 indices separately for keyword search
        self.tfidf_vectorizers = {}  # Store TF-IDF vectorizers
        self.sentence_transformers = {}  # Store sentence transformer models
        
        # Initialize Qdrant client
        if qdrant_path is None:
            self.qdrant_path = tempfile.mkdtemp(prefix="qdrant_db_")
            self.temp_dir = True
        else:
            self.qdrant_path = qdrant_path
            Path(qdrant_path).mkdir(parents=True, exist_ok=True)
            self.temp_dir = False
        
        self.client = QdrantClient(path=self.qdrant_path)
        
        # Determine if single or multiple databases
        if isinstance(texts[0], str):
            texts = [texts]
        
        # Set up database names
        if db_names is None:
            self.db_names = [f"db_{i}" for i in range(len(texts))]
        else:
            self.db_names = db_names[:len(texts)]
        
        # Create chunk configuration
        chunk_config = ChunkConfig(
            chunk_size=chunk_size,
            overlap=overlap,
            chunking_method=chunking_method
        )
        
        # Initialize databases
        for i, text_list in enumerate(texts):
            db_config = DatabaseConfig(
                name=self.db_names[i],
                embedding_model=embedding_model,
                chunk_config=chunk_config
            )
            self._create_database(i, text_list, db_config)
        
        print(f"Initialized {len(self.collections)} database(s) with Qdrant backend")
        for i, name in enumerate(self.db_names):
            collection_info = self.client.get_collection(self.collections[i])
            print(f"  DB {i} ({name}): {collection_info.points_count} points")
    
    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_dir') and self.temp_dir and hasattr(self, 'qdrant_path'):
            try:
                shutil.rmtree(self.qdrant_path)
            except:
                pass
    
    def _chunk_text(self, text: str, config: ChunkConfig) -> List[str]:
        """Chunk text according to configuration"""
        
        if config.chunking_method == "sliding_window":
            words = text.split()
            chunks = []
            for i in range(0, len(words), config.chunk_size - config.overlap):
                chunk_words = words[i:i + config.chunk_size]
                if chunk_words:
                    chunks.append(" ".join(chunk_words))
            return chunks
        
        elif config.chunking_method == "sentence":
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length > config.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Keep overlap sentences
                    overlap_words = config.overlap
                    while overlap_words > 0 and current_chunk:
                        last_sentence = current_chunk.pop()
                        overlap_words -= len(last_sentence.split())
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks
        
        elif config.chunking_method == "paragraph":
            paragraphs = text.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            chunks = []
            current_chunk = []
            current_length = 0
            
            for paragraph in paragraphs:
                paragraph_length = len(paragraph.split())
                if current_length + paragraph_length > config.chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [paragraph]
                    current_length = paragraph_length
                else:
                    current_chunk.append(paragraph)
                    current_length += paragraph_length
            
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            return chunks
        
        else:
            raise ValueError(f"Unknown chunking method: {config.chunking_method}")
    
    def _create_embeddings(self, chunks: List[str], model: str, db_index: int) -> Tuple[np.ndarray, int]:
        """Create embeddings for chunks and return embeddings with dimension"""
        
        if model == "tfidf":
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            embeddings = vectorizer.fit_transform(chunks).toarray()
            self.tfidf_vectorizers[db_index] = vectorizer
            return embeddings, embeddings.shape[1]
        
        elif model == "openai":
            try:
                import openai
                print("OpenAI embeddings require API key and implementation. Falling back to TF-IDF")
                return self._create_embeddings(chunks, "tfidf", db_index)
            except ImportError:
                print("OpenAI package not installed. Falling back to TF-IDF")
                return self._create_embeddings(chunks, "tfidf", db_index)
        
        elif model == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
                # Use provided model name if available, else fall back to default
                model_name = getattr(self, 'sentence_transformer_model_name', 'all-MiniLM-L6-v2')
                model_obj = SentenceTransformer(model_name)
                embeddings = model_obj.encode(chunks, show_progress_bar=True)
                self.sentence_transformers[db_index] = model_obj
                return embeddings, embeddings.shape[1]
            except ImportError:
                print("sentence-transformers not installed. Falling back to TF-IDF")
                return self._create_embeddings(chunks, "tfidf", db_index)
            
        else:
            raise ValueError(f"Unknown embedding model: {model}")
    
    def _create_database(self, db_index: int, texts: List[str], config: DatabaseConfig):
        """Create a single database using Qdrant"""
        
        # Create collection name
        collection_name = f"collection_{db_index}_{uuid.uuid4().hex[:8]}"
        self.collections[db_index] = collection_name
        
        # Chunk all texts
        all_chunks = []
        chunk_metadata = []
        
        for doc_id, text in enumerate(texts):
            chunks = self._chunk_text(text, config.chunk_config)
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'original_text': text[:100] + "..." if len(text) > 100 else text,
                    'db_index': db_index
                })
        
        # Create embeddings
        # Use resolved embedding model type for creating embeddings
        embeddings, vector_size = self._create_embeddings(all_chunks, self.embedding_model_type, db_index)
        
        # Create Qdrant collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        # Insert points into Qdrant
        points = []
        for i, (chunk, metadata, embedding) in enumerate(zip(all_chunks, chunk_metadata, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    'chunk': chunk,
                    'doc_id': metadata['doc_id'],
                    'chunk_id': metadata['chunk_id'],
                    'original_text': metadata['original_text'],
                    'db_index': metadata['db_index']
                }
            )
            points.append(point)
        
        # Batch insert for better performance
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)
        
        # Create BM25 index for keyword search
        self.bm25_indices[db_index] = BM25(all_chunks)
        
        print(f"Created database {db_index} with {len(all_chunks)} chunks in Qdrant collection: {collection_name}")
    
    def search(self, 
               query: str, 
               method: str = "embeddings",
               k: int = 5,
               db_index: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search in the specified database using Qdrant backend
        
        Args:
            query: Search query
            method: Search method ("bm25", "embeddings", "hybrid", "matching")
            k: Number of results to return
            db_index: Database index (uses current_db_index if None)
        
        Returns:
            List of search results with metadata
        """
        
        if db_index is None:
            db_index = self.current_db_index
            
        print(f"Searching in database {db_index} with method {method} and k={k}")
        
        if db_index not in self.collections:
            raise ValueError(f"Database index {db_index} not found")
        
        if method == "bm25":
            return self._search_bm25(query, db_index, k)
        elif method == "embeddings":
            return self._search_embeddings(query, db_index, k)
        elif method == "hybrid":
            return self._search_hybrid(query, db_index, k)
        elif method == "matching":
            return self._search_matching(query, db_index, k)
        else:
            raise ValueError(f"Unknown search method: {method}")
    
    def _search_bm25(self, query: str, db_index: int, k: int) -> List[Dict[str, Any]]:
        """Search using BM25 - uses local BM25 index"""
        if db_index not in self.bm25_indices:
            raise ValueError(f"BM25 index not found for database {db_index}")
        
        results = self.bm25_indices[db_index].search(query, k)
        
        # Get chunk data from Qdrant for the results
        collection_name = self.collections[db_index]
        output = []
        
        for idx, score in results:
            try:
                point = self.client.retrieve(
                    collection_name=collection_name,
                    ids=[idx]
                )[0]
                
                output.append({
                    'chunk': point.payload['chunk'],
                    'score': score,
                    'metadata': {
                        'doc_id': point.payload['doc_id'],
                        'chunk_id': point.payload['chunk_id'],
                        'original_text': point.payload['original_text']
                    },
                    'method': 'bm25'
                })
            except:
                continue  # Skip if point not found
        
        return output
    
    def _search_embeddings(self, query: str, db_index: int, k: int) -> List[Dict[str, Any]]:
        """Search using embeddings similarity via Qdrant"""
        
        collection_name = self.collections[db_index]
        
        # Create query embedding
        if self.embedding_model_type == "tfidf":
            if db_index not in self.tfidf_vectorizers:
                raise ValueError(f"TF-IDF vectorizer not found for database {db_index}")
            query_vec = self.tfidf_vectorizers[db_index].transform([query]).toarray()[0]
        elif self.embedding_model_type == "sentence_transformers":
            if db_index not in self.sentence_transformers:
                raise ValueError(f"Sentence transformer not found for database {db_index}")
            query_vec = self.sentence_transformers[db_index].encode([query])[0]
        else:
            raise ValueError(f"Unsupported embedding model type: {self.embedding_model_type}")
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vec.tolist(),
            limit=k
        )
        
        output = []
        for result in search_results:
            output.append({
                'chunk': result.payload['chunk'],
                'score': result.score,
                'metadata': {
                    'doc_id': result.payload['doc_id'],
                    'chunk_id': result.payload['chunk_id'],
                    'original_text': result.payload['original_text']
                },
                'method': 'embeddings'
            })
        
        return output
    
    def _search_hybrid(self, query: str, db_index: int, k: int) -> List[Dict[str, Any]]:
        """Search using hybrid BM25 + embeddings"""
        
        # Get results from both methods
        bm25_results = self._search_bm25(query, db_index, k * 2)
        emb_results = self._search_embeddings(query, db_index, k * 2)
        
        # Normalize scores
        if bm25_results:
            max_bm25 = max(r['score'] for r in bm25_results)
            for r in bm25_results:
                r['score'] = r['score'] / max_bm25 if max_bm25 > 0 else 0
        
        if emb_results:
            max_emb = max(r['score'] for r in emb_results)
            for r in emb_results:
                r['score'] = r['score'] / max_emb if max_emb > 0 else 0
        
        # Combine results by chunk content
        combined_scores = defaultdict(float)
        result_data = {}
        
        for r in bm25_results:
            chunk = r['chunk']
            combined_scores[chunk] += 0.5 * r['score']  # 50% weight for BM25
            result_data[chunk] = r
        
        for r in emb_results:
            chunk = r['chunk']
            combined_scores[chunk] += 0.5 * r['score']  # 50% weight for embeddings
            if chunk not in result_data:
                result_data[chunk] = r
        
        # Sort by combined score
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        output = []
        for chunk, score in sorted_chunks:
            result = result_data[chunk].copy()
            result['score'] = score
            result['method'] = 'hybrid'
            output.append(result)
        
        return output
    
    def _search_matching(self, query: str, db_index: int, k: int) -> List[Dict[str, Any]]:
        """Search using simple text matching via Qdrant scroll"""
        
        collection_name = self.collections[db_index]
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Use Qdrant scroll to iterate through all points
        results = []
        offset = None
        
        while True:
            points, next_offset = self.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset
            )
            
            if not points:
                break
            
            for point in points:
                chunk = point.payload['chunk']
                chunk_lower = chunk.lower()
                
                # Count query word matches
                matches = sum(1 for word in query_words if word in chunk_lower)
                
                if matches > 0:
                    score = matches / len(query_words)  # Proportion of query words found
                    results.append({
                        'chunk': chunk,
                        'score': score,
                        'metadata': {
                            'doc_id': point.payload['doc_id'],
                            'chunk_id': point.payload['chunk_id'],
                            'original_text': point.payload['original_text']
                        },
                        'method': 'matching',
                        'matches': matches
                    })
            
            offset = next_offset
            if offset is None:
                break
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def set_default_db(self, db_index: int):
        """Set the default database index"""
        if db_index not in self.collections:
            raise ValueError(f"Database index {db_index} not found")
        self.current_db_index = db_index
        print(f"Default database set to {db_index} ({self.db_names[db_index]})")
    
    def list_databases(self):
        """List all databases with their info"""
        print(f"Total databases: {len(self.collections)}")
        print(f"Current default: {self.current_db_index} ({self.db_names[self.current_db_index]})")
        print(f"Qdrant storage path: {self.qdrant_path}")
        print("\nDatabases:")
        for i in self.collections.keys():
            collection_name = self.collections[i]
            collection_info = self.client.get_collection(collection_name)
            print(f"  [{i}] {self.db_names[i]}")
            print(f"      Collection: {collection_name}")
            print(f"      Points: {collection_info.points_count}")
            print(f"      Vector size: {collection_info.config.params.vectors.size}")
            print(f"      Distance: {collection_info.config.params.vectors.distance}")
    
    def get_database_stats(self, db_index: Optional[int] = None) -> Dict[str, Any]:
        """Get detailed statistics for a database"""
        if db_index is None:
            db_index = self.current_db_index
        
        if db_index not in self.collections:
            raise ValueError(f"Database index {db_index} not found")
        
        collection_name = self.collections[db_index]
        collection_info = self.client.get_collection(collection_name)
        
        return {
            'db_index': db_index,
            'db_name': self.db_names[db_index],
            'collection_name': collection_name,
            'points_count': collection_info.points_count,
            'vector_size': collection_info.config.params.vectors.size,
            'distance_metric': collection_info.config.params.vectors.distance,
            'embedding_model': self.embedding_model
        }
    
    def save(self, filepath: str):
        """Save the database configuration and create a backup"""
        
        # Create save data (configuration only)
        save_data = {
            'embedding_model': self.embedding_model,
            'embedding_model_type': getattr(self, 'embedding_model_type', None),
            'current_db_index': self.current_db_index,
            'db_names': self.db_names,
            'collections': self.collections,
            'qdrant_path': self.qdrant_path,
            'database_stats': {}
        }
        
        # Save database statistics
        for db_index in self.collections.keys():
            save_data['database_stats'][db_index] = self.get_database_stats(db_index)
        
        # Save TF-IDF vectorizers
        tfidf_models = {}
        for db_idx, vectorizer in self.tfidf_vectorizers.items():
            model_path = f"{filepath}_tfidf_{db_idx}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            tfidf_models[db_idx] = model_path
        
        save_data['tfidf_models'] = tfidf_models
        # Persist the sentence-transformers model name actually used
        save_data['sentence_transformer_model'] = getattr(self, 'sentence_transformer_model_name', 'all-MiniLM-L6-v2')
        
        # Save configuration
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Create backup of Qdrant data
        backup_path = f"{filepath}_qdrant_backup"
        try:
            shutil.copytree(self.qdrant_path, backup_path, dirs_exist_ok=True)
            save_data['qdrant_backup_path'] = backup_path
            
            # Re-save with backup path
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"Database saved to {filepath}")
            print(f"Qdrant backup created at {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create Qdrant backup: {e}")
            print(f"Configuration saved to {filepath} (without Qdrant backup)")
    
    @classmethod
    def load(cls, filepath: str, new_qdrant_path: Optional[str] = None) -> 'QdrantVectorDB':
        """Load database from saved configuration"""
        
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required. Install with: pip install qdrant-client")
        
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        # Determine Qdrant path
        if new_qdrant_path:
            qdrant_path = new_qdrant_path
            Path(qdrant_path).mkdir(parents=True, exist_ok=True)
        elif 'qdrant_backup_path' in save_data and os.path.exists(save_data['qdrant_backup_path']):
            qdrant_path = tempfile.mkdtemp(prefix="qdrant_db_restored_")
            shutil.copytree(save_data['qdrant_backup_path'], qdrant_path, dirs_exist_ok=True)
        else:
            raise FileNotFoundError("No Qdrant backup found and no new path specified")
        
        # Create new instance
        instance = cls.__new__(cls)
        instance.embedding_model = save_data['embedding_model']
        instance.embedding_model_type = save_data.get('embedding_model_type', 'tfidf' if save_data['embedding_model'] == 'tfidf' else 'sentence_transformers')
        instance.current_db_index = save_data['current_db_index']
        instance.db_names = save_data['db_names']
        instance.collections = {int(k): v for k, v in save_data['collections'].items()}
        instance.qdrant_path = qdrant_path
        instance.temp_dir = new_qdrant_path is None
        
        # Initialize Qdrant client
        instance.client = QdrantClient(path=qdrant_path)
        
        # Load TF-IDF vectorizers
        instance.tfidf_vectorizers = {}
        if 'tfidf_models' in save_data:
            for db_idx_str, model_path in save_data['tfidf_models'].items():
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        instance.tfidf_vectorizers[int(db_idx_str)] = pickle.load(f)
        
        # Load sentence transformers
        instance.sentence_transformers = {}
        if instance.embedding_model_type == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
                model_name = save_data.get('sentence_transformer_model', 'all-MiniLM-L6-v2')
                instance.sentence_transformer_model_name = model_name
                for db_idx in instance.collections.keys():
                    instance.sentence_transformers[db_idx] = SentenceTransformer(model_name)
                print(f"Recreated SentenceTransformer models: {model_name}")
            except ImportError:
                print("sentence-transformers not available. Embedding search may not work.")
        
        # Recreate BM25 indices from Qdrant data
        instance.bm25_indices = {}
        for db_idx, collection_name in instance.collections.items():
            chunks = []
            offset = None
            
            while True:
                points, next_offset = instance.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset
                )
                
                if not points:
                    break
                
                for point in points:
                    chunks.append(point.payload['chunk'])
                
                offset = next_offset
                if offset is None:
                    break
            
            instance.bm25_indices[db_idx] = BM25(chunks)
        
        print(f"Database loaded from {filepath}")
        print(f"Qdrant restored to: {qdrant_path}")
        instance.list_databases()
        
        return instance
    
    def delete_database(self, db_index: int):
        """Delete a specific database"""
        if db_index not in self.collections:
            raise ValueError(f"Database index {db_index} not found")
        
        collection_name = self.collections[db_index]
        
        # Delete Qdrant collection
        self.client.delete_collection(collection_name)
        
        # Clean up associated data
        del self.collections[db_index]
        if db_index in self.bm25_indices:
            del self.bm25_indices[db_index]
        if db_index in self.tfidf_vectorizers:
            del self.tfidf_vectorizers[db_index]
        if db_index in self.sentence_transformers:
            del self.sentence_transformers[db_index]
        
        # Update current db if necessary
        if db_index == self.current_db_index and self.collections:
            self.current_db_index = min(self.collections.keys())
        
        print(f"Database {db_index} deleted successfully")


# Example usage and testing
if __name__ == "__main__":
    # Sample documents
    docs1 = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information from images."
    ]
    
    docs2 = [
        "Python is a popular programming language for data science and machine learning.",
        "JavaScript is widely used for web development and frontend applications.",
        "SQL is essential for database management and data retrieval.",
        "Git is a version control system that tracks changes in source code."
    ]
    
    # Test single database with Qdrant
    print("=== Single Database Test with Qdrant ===")
    try:
        single_db = QdrantVectorDB(
            texts=docs1,
            embedding_model="tfidf",
            chunk_size=100,
            overlap=20
        )
        
        # Test different search methods
        print("\n--- Testing BM25 Search ---")
        results = single_db.search("neural networks", method="bm25", k=2)
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.3f}")
            print(f"   Text: {result['chunk'][:100]}...")
            print(f"   Method: {result['method']}")
        
        print("\n--- Testing Embedding Search ---")
        results = single_db.search("neural networks", method="embeddings", k=2)
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.3f}")
            print(f"   Text: {result['chunk'][:100]}...")
            print(f"   Method: {result['method']}")
        
        print("\n--- Testing Hybrid Search ---")
        results = single_db.search("neural networks", method="hybrid", k=2)
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.3f}")
            print(f"   Text: {result['chunk'][:100]}...")
            print(f"   Method: {result['method']}")
        
    except ImportError as e:
        print(f"Qdrant not available: {e}")
        print("Install with: pip install qdrant-client")
    
    # Test multiple databases
    print("\n\n=== Multiple Database Test with Qdrant ===")
    try:
        multi_db = QdrantVectorDB(
            texts=[docs1, docs2],
            embedding_model="sentence_transformers",  # Test with sentence transformers
            chunk_size=100,
            overlap=20,
            db_names=["AI_docs", "Programming_docs"]
        )
        
        multi_db.list_databases()
        
        # Search in different databases
        print(f"\nSearch in DB 0 (AI):")
        results = multi_db.search("machine learning", method="bm25", k=2, db_index=0)
        for result in results:
            print(f"- {result['chunk'][:80]}... (Score: {result['score']:.3f})")
        
        print(f"\nSearch in DB 1 (Programming):")
        results = multi_db.search("programming language", method="hybrid", k=2, db_index=1)
        for result in results:
            print(f"- {result['chunk'][:80]}... (Score: {result['score']:.3f})")
        
        # Test database statistics
        print(f"\n--- Database Statistics ---")
        for db_idx in [0, 1]:
            stats = multi_db.get_database_stats(db_idx)
            print(f"DB {db_idx}: {stats['points_count']} points, "
                  f"vector dim: {stats['vector_size']}, "
                  f"distance: {stats['distance_metric']}")
        
        # Test save/load
        print(f"\n=== Save/Load Test ===")
        multi_db.save("test_qdrant_db.json")
        
        # Load database
        print("\n--- Loading Database ---")
        loaded_db = QdrantVectorDB.load("test_qdrant_db.json")
        
        # Test loaded database
        results = loaded_db.search("neural networks", method="embeddings", k=1)
        print(f"\nLoaded DB search result:")
        if results:
            print(f"- {results[0]['chunk'][:80]}... (Score: {results[0]['score']:.3f})")
        
        # Test matching search
        print(f"\n--- Testing Text Matching Search ---")
        results = loaded_db.search("machine learning", method="matching", k=2, db_index=0)
        for result in results:
            print(f"- {result['chunk'][:60]}... "
                  f"(Score: {result['score']:.3f}, Matches: {result['matches']})")
        
    except ImportError as e:
        print(f"Required packages not available: {e}")
        print("Install with: pip install qdrant-client sentence-transformers")
    except Exception as e:
        print(f"Error in multiple database test: {e}")
        import traceback
        traceback.print_exc()