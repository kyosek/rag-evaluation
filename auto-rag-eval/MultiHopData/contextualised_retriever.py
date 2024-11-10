import os
import json
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import random
from tqdm import tqdm

from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.gcp.gemini_instant import GeminiGcp


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    original_index: int
    context: str = ""

class ContextualChunkRetriever:
    def __init__(self, task_domain: str, model_name: str = 'all-MiniLM-L6-v2', random_seed: Optional[int] = None):
        self.model = SentenceTransformer(model_name)
        self.task_domain = task_domain
        self.index = None
        self.chunks: List[Chunk] = []
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
        
        self.claude = ClaudeGcp(model_name="claude-3-5-haiku@20241022")
        self.gemini = GeminiGcp(model_name="gemini-1.5-flash-002")

    def generate_context(self, doc_content: str, chunk_content: str, model_name) -> str:
        prompt = f'''
        <document>
        {doc_content}
        </document>

        <chunk>
        {chunk_content}
        </chunk>

        Create a concise a few sentences summary that:
        1. Describes how this chunk fits into the broader document
        2. Captures key relationships to other document sections
        3. Identifies the main technical concepts/terms
        4. Notes any dependencies or prerequisites mentioned
        5. Highlights unique identifying details

        Focus on information that would help distinguish this chunk from similar content.
        Be specific but concise.
        Avoid generic descriptions. Include technical terms that someone might search for.

        Output only the contextual summary, with no additional text or explanations.
        '''
        llm = model_name

        response = llm.invoke(prompt)
        return response.strip()

    def load_documents(self, json_file: str) -> None:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for doc in tqdm(data, desc="Processing documents"):
            doc_id = doc['doc_id']
            doc_content = "\n".join([chunk['content'] for chunk in doc['chunks']])
            
            for chunk in doc['chunks']:
                try:
                    context = self.generate_context(doc_content, chunk['content'], self.gemini)
                except:
                    print("Generating with Claude")
                    context = self.generate_context(doc_content, chunk['content'], self.claude)
                chunk_obj = Chunk(
                    chunk_id=chunk['chunk_id'],
                    doc_id=doc_id,
                    content=chunk['content'],
                    original_index=chunk['original_index'],
                    context=context
                )
                self.chunks.append(chunk_obj)
        
        self._build_index()

    def _build_index(self) -> None:
        embeddings = self.model.encode([f"{chunk.content}\n\nContext: {chunk.context}" for chunk in self.chunks])
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def sample_chunks(self, n: int, seed: Optional[int] = None) -> List[Chunk]:
        if seed is not None:
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
    ) -> List[tuple[Chunk, float]]:
        query_embedding = self.model.encode([f"{query_chunk.content}\n\nContext: {query_chunk.context}"])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k * 2)
        
        similar_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score < similarity_threshold:
                continue
            
            chunk = self.chunks[idx]
            if exclude_same_doc and chunk.doc_id == query_chunk.doc_id:
                continue
            
            if chunk.chunk_id != query_chunk.chunk_id:
                similar_chunks.append((chunk, float(score)))
            
            if len(similar_chunks) >= k:
                break
        
        return similar_chunks

    def save_database(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
        
        metadata = {
            'chunks': self.chunks,
            'random_seed': self.random_seed
        }
        with open(os.path.join(directory, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)

    @classmethod
    def load_database(cls, directory: str, task_domain: str, model_name: str = 'all-MiniLM-L6-v2') -> 'ContextualChunkRetriever':
        with open(os.path.join(directory, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
        
        instance = cls(task_domain, model_name=model_name, random_seed=metadata['random_seed'])
        instance.chunks = metadata['chunks']
        
        instance.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
        
        if instance.index is None:
            instance._build_index()
        
        return instance

def main(task_domain: str):
    chunk_retriever = ContextualChunkRetriever(task_domain, random_seed=42)
    chunk_retriever.load_documents(f"MultiHopData/{task_domain}/docs_chunk.json")
    chunk_retriever.save_database(f"MultiHopData/{task_domain}/contextual_chunk_flash_database")


if __name__ == "__main__":
    # task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    task_domains = ["hotpotqa"]
    # task_domains = ["gov_report", "hotpotqa", "multifieldqa_en"]
    for task_domain in task_domains:
        print(f"Processing {task_domain}")
        main(task_domain)
