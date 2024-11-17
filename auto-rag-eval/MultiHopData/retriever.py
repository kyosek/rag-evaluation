from abc import ABC, abstractmethod
import json
import faiss
import random
import os
import pickle
import numpy as np

from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from sentence_transformers import CrossEncoder


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    original_index: int


class BaseRetriever(ABC):
    """Abstract base class for different retrieval methods."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query."""
        pass


class ChunkRetriever:
    def __init__(
        self,
        task_domain: str,
        model_name: str = "all-MiniLM-L6-v2",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the chunk retriever with a sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model to use
            random_seed: Seed for random operations (optional)
        """
        self.model = SentenceTransformer(model_name)
        self.task_domain = task_domain
        self.index = None
        self.chunks: List[Chunk] = []
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

    def load_documents(self, json_file: str) -> None:
        """
        Load documents from JSON file and store chunks.

        Args:
            json_file: Path to the JSON file containing documents
        """
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract all chunks from the documents
        for doc in data:
            doc_id = doc["doc_id"]
            for chunk in doc["chunks"]:
                chunk_obj = Chunk(
                    chunk_id=chunk["chunk_id"],
                    doc_id=doc_id,
                    content=chunk["content"],
                    original_index=chunk["original_index"],
                )
                self.chunks.append(chunk_obj)

        # Create FAISS index
        self._build_index()

    def _build_index(self) -> None:
        """Build FAISS index from chunks."""
        # Generate embeddings for all chunks
        embeddings = self.model.encode([chunk.content for chunk in self.chunks])

        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(
            dimension
        )  # Inner product is equivalent to cosine similarity for normalized vectors

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add vectors to the index
        self.index.add(embeddings)

    def sample_chunks(self, n: int, seed: Optional[int] = None) -> List[Chunk]:
        """
        Randomly sample n chunks from the dataset.

        Args:
            n: Number of chunks to sample
            seed: Random seed for sampling (overrides instance seed if provided)

        Returns:
            List of sampled chunks
        """
        if seed is not None:
            # Temporarily set seed for this operation
            current_state = random.getstate()
            random.seed(seed)
            samples = random.sample(self.chunks, min(n, len(self.chunks)))
            random.setstate(current_state)
            return samples
        return random.sample(self.chunks, min(n, len(self.chunks)))

    def find_similar_chunks(
        self,
        query_chunk: Chunk,
        k: int = 4,
        similarity_threshold: float = 0.9,
        exclude_same_doc: bool = True,
    ) -> List[Tuple[Chunk, float]]:
        """
        Find similar chunks for a given query chunk.

        Args:
            query_chunk: The chunk to find similar chunks for
            k: Number of similar chunks to retrieve
            similarity_threshold: Minimum similarity score threshold
            exclude_same_doc: Whether to exclude chunks from the same document

        Returns:
            List of tuples containing similar chunks and their similarity scores
        """
        # Generate embedding for query chunk
        query_embedding = self.model.encode([query_chunk.content])
        faiss.normalize_L2(query_embedding)

        # Search in the index
        scores, indices = self.index.search(
            query_embedding, k * 2
        )  # Get more results initially for filtering

        # Filter and process results
        similar_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score < similarity_threshold:
                continue

            chunk = self.chunks[idx]
            if exclude_same_doc and chunk.doc_id == query_chunk.doc_id:
                continue

            if chunk.chunk_id != query_chunk.chunk_id:  # Exclude the query chunk itself
                similar_chunks.append((chunk, float(score)))

            if len(similar_chunks) >= k:
                break

        return similar_chunks

    def save_database(self, directory: str) -> None:
        """
        Save the database (FAISS index and chunks) to disk.

        Args:
            directory: Directory to save the database files
        """
        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))

        # Save chunks and other metadata
        metadata = {"chunks": self.chunks, "random_seed": self.random_seed}
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load_database(
        cls, directory: str, task_domain: str, model_name: str = "all-MiniLM-L6-v2"
    ) -> "ChunkRetriever":
        """
        Load a previously saved database.

        Args:
            directory: Directory containing the saved database files
            task_domain: Domain of the task
            model_name: Name of the sentence transformer model to use

        Returns:
            ChunkRetriever instance with loaded data
        """
        # Load metadata
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(task_domain, model_name=model_name, random_seed=metadata["random_seed"])
        instance.chunks = metadata["chunks"]

        # Load FAISS index
        instance.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))

        # Make sure the index is properly loaded
        if instance.index is None:
            # Rebuild index if loading failed
            instance._build_index()

        return instance


class HybridChunkRetriever(ChunkRetriever):
    def __init__(
        self,
        task_domain: str,
        bi_encoder_name: str = "all-MiniLM-L6-v2",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the hybrid chunk retriever with both bi-encoder and cross-encoder models.
        """
        super().__init__(task_domain, model_name=bi_encoder_name, random_seed=random_seed)
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        self.cross_encoder_name = cross_encoder_name

    def load_documents(self, json_file: str) -> None:
        """
        Load documents from JSON file and store chunks.

        Args:
            json_file: Path to the JSON file containing documents
        """
        print(f"Loading documents from {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Found {len(data)} documents")
        
        # Clear existing chunks if any
        self.chunks = []
        
        # Extract all chunks from the documents
        total_chunks = sum(len(doc["chunks"]) for doc in data)
        print(f"Processing {total_chunks} chunks...")
        
        for doc in tqdm(data, desc="Processing documents"):
            doc_id = doc["doc_id"]
            for chunk in doc["chunks"]:
                chunk_obj = Chunk(
                    chunk_id=chunk["chunk_id"],
                    doc_id=doc_id,
                    content=chunk["content"],
                    original_index=chunk["original_index"],
                )
                self.chunks.append(chunk_obj)
        
        print("Building FAISS index...")
        self._build_index()
        print("Finished loading documents and building index")

    def _build_index(self) -> None:
        """Build FAISS index from chunks with progress bar."""
        # Generate embeddings for all chunks with progress bar
        print("Generating embeddings...")
        embeddings = []
        batch_size = 32  # Adjust based on your memory constraints
        
        for i in tqdm(range(0, len(self.chunks), batch_size), desc="Encoding chunks"):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = self.model.encode([chunk.content for chunk in batch])
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Initialize FAISS index
        print("Initializing FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize vectors for cosine similarity
        print("Normalizing vectors...")
        faiss.normalize_L2(embeddings)
        
        # Add vectors to the index
        print("Adding vectors to index...")
        self.index.add(embeddings)
        
        print(f"Index built successfully with {len(self.chunks)} vectors")
        
    def find_similar_chunks(
        self,
        query_chunk: Chunk,
        k: int = 4,
        initial_k: int = 20,
        similarity_threshold: float = 0.5,
        exclude_same_doc: bool = True,
        batch_size: int = 32,
    ) -> List[Tuple[Chunk, float]]:
        """
        Find similar chunks using a hybrid approach with batched processing.
        """
        # Step 1: Initial retrieval with bi-encoder (FAISS)
        query_embedding = self.model.encode([query_chunk.content])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, initial_k)
        
        # Prepare candidates for cross-encoder
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[idx]
            if exclude_same_doc and chunk.doc_id == query_chunk.doc_id:
                continue
            if chunk.chunk_id == query_chunk.chunk_id:
                continue
            candidates.append((chunk, float(score)))
        
        if not candidates:
            return []
            
        # Step 2: Re-rank with cross-encoder using batched processing
        candidate_chunks = [c[0] for c in candidates]
        
        # Prepare pairs for cross-encoder
        pairs = [(query_chunk.content, chunk.content) for chunk in candidate_chunks]
        
        # Process in batches
        cross_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = self.cross_encoder.predict(batch)
            if isinstance(batch_scores, (float, np.float32, np.float64)):
                batch_scores = [batch_scores]
            cross_scores.extend(batch_scores)
        
        # Combine results
        reranked_results = list(zip(candidate_chunks, cross_scores))
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and take top k
        final_results = [
            (chunk, score) 
            for chunk, score in reranked_results 
            # if score >= similarity_threshold
        ][:k]
        
        return final_results

    def save_database(self, directory: str) -> None:
        """
        Save the database including both encoder models.
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        print(f"Saving FAISS index to {directory}...")
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
        
        # Save chunks and other metadata
        print("Saving metadata...")
        metadata = {
            "chunks": self.chunks,
            "random_seed": self.random_seed
        }
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
            
        # Save cross-encoder configuration
        hybrid_metadata = {
            "cross_encoder_name": self.cross_encoder_name
        }
        with open(os.path.join(directory, "hybrid_metadata.json"), "w") as f:
            json.dump(hybrid_metadata, f)
            
        print(f"Database saved successfully to {directory}")

    @classmethod
    def load_database(
        cls,
        directory: str,
        task_domain: str,
        bi_encoder_name: str = "all-MiniLM-L6-v2",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> "HybridChunkRetriever":
        """
        Load a previously saved database.
        """
        print(f"Loading database from {directory}...")
        
        # Load cross-encoder configuration
        with open(os.path.join(directory, "hybrid_metadata.json"), "r") as f:
            hybrid_metadata = json.load(f)
            cross_encoder_name = hybrid_metadata.get("cross_encoder_name", cross_encoder_name)
        
        # Create instance
        print("Initialising models...")
        instance = cls(
            task_domain=task_domain,
            bi_encoder_name=bi_encoder_name,
            cross_encoder_name=cross_encoder_name
        )
        
        # Load metadata
        print("Loading metadata...")
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        instance.chunks = metadata["chunks"]
        
        # Load FAISS index
        print("Loading FAISS index...")
        instance.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
        
        print(f"Database loaded successfully with {len(instance.chunks)} chunks")
        return instance


class RerankingRetriever(BaseRetriever):
    def __init__(
        self,
        base_retriever: BaseRetriever,
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.base_retriever = base_retriever
        self.rerank_model = CrossEncoder(rerank_model_name)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        # Get initial results
        initial_results = self.base_retriever.retrieve(query, k=k * 2)  # Retrieve more initially

        # Check if we have any results
        if not initial_results:
            print(f"Warning: No results found for query: {query}")
            return []

        # Prepare pairs for re-ranking
        pairs = [[query, doc] for doc, _ in initial_results]

        # Re-rank
        scores = self.rerank_model.predict(pairs)

        # Sort and return top k
        reranked_results = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)[
            :k
        ]
        return [(doc, score) for (doc, _), score in reranked_results]
