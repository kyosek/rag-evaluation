import json
import numpy as np
import faiss
import random
import os
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import pickle
from tqdm import tqdm

# from LLMServer.base_model import BaseLLM
from LLMServer.gcp.claude_instant import ClaudeGcp

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
        k: int = 4, 
        similarity_threshold: float = 0.9,
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


def generate_exam(data: List[Dict[str, str]], step_size: int, task_domain: str, retriever: ChunkRetriever, llm_model) -> List[Dict[str, str]]:
    """
    Generate an exam with multiple-choice questions from the given data.
    
    Args:
        data: List of dictionaries containing the documentation for each chunk
        step_size: Number of chunks to process at a time
        task_domain: Domain of the documentation
        llm_model: Language model to use for generating questions
        
    Returns:
        List of dictionaries representing the generated exam questions
    """
    exam = []
    
    for k in tqdm(range(0, len(data), step_size)):
        # Get the current chunk and its similar chunks
        current_chunk = data[k]
        similar_chunks = retriever.find_similar_chunks(
            Chunk(
                chunk_id=current_chunk["chunk_id"],
                doc_id=current_chunk["doc_id"],
                content=current_chunk["text"],
                original_index=k
            ),
            k=4,
            similarity_threshold=0.9,
            exclude_same_doc=True
        )
        similar_chunk_data = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.content} for c, _ in similar_chunks]
        
        # Generate a high-level (L3) question
        question_prompt = make_l3_question_prompt(task_domain, similar_chunk_data)
        answer = llm_model.invoke(prompt=question_prompt, params={})
        
        # Extract question, choices, correct answer, and explanation
        question = answer.split("\nQuestion: ")[1].split("\nA)")[0]
        choices = answer.split("\nA)")[1].split("\nB)")[1].split("\nC)")[1].split("\nD)")[1].split("\nCorrect Answer:")[0].split("\n")
        correct_answer = answer.split("\nCorrect Answer: ")[1].split("\nExplanation:")[0]
        explanation = answer.split("\nExplanation: ")[1]
        
        # Construct the exam entry
        exam_entry = {
            "question": question,
            "documentation": [chunk["text"] for chunk in similar_chunk_data],
            "choices": [choice.strip() for choice in choices],
            "correct_answer": f"{correct_answer}) {explanation}"
        }
        
        exam.append(exam_entry)
    
    return exam


def make_l3_question_prompt(task_domain: str, chunks: List[Dict[str, str]]) -> str:
    documentation = "\n\n".join([chunk["text"] for chunk in chunks])
    return f"""
    <<SYS>>
    You are an expert exam question and answer generator specializing in creating high-quality, challenging multiple-choice questions. 
    Your questions should:
    1. Target L3 (Analysis/Application) or higher cognitive levels in Bloom's taxonomy
    2. Require integration of multiple concepts from the documentation
    3. Test critical thinking rather than memorization
    4. Have carefully crafted distractors that represent common misconceptions
    5. Use all the provided information from the document to generate a question

    Guidelines for creating options:
    - All options should be of similar length and complexity
    - Avoid obvious wrong answers
    - Use common misconceptions as distractors
    - Ensure options are mutually exclusive
    - Avoid "all/none of the above" options
    <</SYS>>

    Domain: {task_domain}
    Documentation: {documentation}

    Example questions based on different domains:

    Technical Documentation Example:
    Question: A microservice is experiencing intermittent failures during peak load. Given the following error logs and system metrics, what is the most likely root cause?
    A) Network timeout due to connection pool exhaustion
    B) Memory leak in the application container
    C) Database connection throttling
    D) CPU throttling by the container orchestrator
    Correct Answer: A
    Explanation: The logs show increasing connection wait times...

    Medical Guidelines Example:
    Question: A 45-year-old patient presents with acute chest pain (7/10), radiating to the left arm, with associated shortness of breath. The ECG shows ST-segment elevation in leads V1-V4. Based on the current guidelines, what is the most appropriate immediate management?
    A) Administer aspirin and arrange urgent PCI
    B) Start thrombolysis and transfer to nearest cardiac center
    C) Perform bedside echocardiogram before any intervention
    D) Administer morphine and arrange CT coronary angiogram
    Correct Answer: A
    Explanation: Given the STEMI presentation...

    Please generate a question following this format:
    Question: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [Letter]
    Explanation: [Detailed explanation]
    """


def main(data_path: str, output_path: str, task_domain: str, step_size: int):
    print("Start processing")
    retriever = ChunkRetriever(random_seed=42)
    model = ClaudeGcp()
    
    # Load documents
    retriever.load_documents(data_path)
    
    # Generate the exam
    exam = generate_exam(retriever.chunks, step_size, task_domain, retriever, model)
    
    # Save the exam to a JSON file
    with open(output_path, "w") as f:
        json.dump(exam, f, indent=2)


if __name__ == "__main__":
    task_domain = "SecFilings"
    data_path = f"auto-rag-eval/MultiHopData/{task_domain}/docs_chunk.json"
    output_path = f"auto-rag-eval/MultiHopData/{task_domain}/exam.json"
    sample_size = 10
    
    main(data_path, output_path, task_domain, sample_size)
