import json
import numpy as np
import faiss
import random
import os
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import pickle

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    original_index: int

class ChunkRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', random_seed: Optional[int] = None):
        """
        Initialize the chunk retriever with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            random_seed: Seed for random operations (optional)
        """
        self.model = SentenceTransformer(model_name)
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
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Extract all chunks from the documents
        for doc in data:
            doc_id = doc['doc_id']
            for chunk in doc['chunks']:
                chunk_obj = Chunk(
                    chunk_id=chunk['chunk_id'],
                    doc_id=doc_id,
                    content=chunk['content'],
                    original_index=chunk['original_index']
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
        self.index = faiss.IndexFlatIP(dimension)  # Inner product is equivalent to cosine similarity for normalized vectors
        
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
        k: int = 5, 
        similarity_threshold: float = 0.8,
        exclude_same_doc: bool = True
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
        scores, indices = self.index.search(query_embedding, k * 2)  # Get more results initially for filtering
        
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
        metadata = {
            'chunks': self.chunks,
            'random_seed': self.random_seed
        }
        with open(os.path.join(directory, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
            
    @classmethod
    def load_database(cls, directory: str, model_name: str = 'all-MiniLM-L6-v2') -> 'ChunkRetriever':
        """
        Load a previously saved database.
        
        Args:
            directory: Directory containing the saved database files
            model_name: Name of the sentence transformer model to use
            
        Returns:
            ChunkRetriever instance with loaded data
        """
        # Load metadata
        with open(os.path.join(directory, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
            
        # Create instance
        instance = cls(model_name=model_name, random_seed=metadata['random_seed'])
        instance.chunks = metadata['chunks']
        
        # Load FAISS index
        instance.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
        
        return instance

def main(data_path: str, sample_size: int = 1000):
    # Initialize the retriever with a fixed seed
    retriever = ChunkRetriever(random_seed=42)
    
    # Load documents
    retriever.load_documents(data_path)
    
    # Save the database
    retriever.save_database('chunk_database')
    
    # Later, you can load the database like this:
    loaded_retriever = ChunkRetriever.load_database('chunk_database')
    
    # Sample 5 random chunks with a specific seed
    sampled_chunks = loaded_retriever.sample_chunks(sample_size, seed=42)
    
    # Process each sampled chunk
    for chunk in sampled_chunks:
        print(f"\nProcessing chunk: {chunk.chunk_id}")
        print(f"Content: {chunk.content[:100]}...")  # Show first 100 characters
        
        # Find similar chunks
        similar_chunks = loaded_retriever.find_similar_chunks(
            chunk,
            k=5,
            similarity_threshold=0.8,
            exclude_same_doc=True
        )
        
        # Print similar chunks
        print("\nSimilar chunks:")
        for similar_chunk, score in similar_chunks:
            print(f"- {similar_chunk.chunk_id} (score: {score:.3f})")
            print(f"  Content: {similar_chunk.content[:100]}...")


if __name__ == "__main__":
    data_path = "auto-rag-eval/MultiHopData/SecFilings/docs_chunk.json"
    sample_size = 10
    
    main(data_path, sample_size)
