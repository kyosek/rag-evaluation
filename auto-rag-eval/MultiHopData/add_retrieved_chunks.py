from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
import nltk
from nltk.tokenize import word_tokenize

from MultiHopData.retriever import BaseRetriever, BM25Retriever, Chunk, ChunkRetriever, FAISSRetriever, HybridRetriever, RerankingRetriever

nltk.download("punkt_tab")


def add_retrieved_chunks_to_exam(
    exam_path: str,
    chunk_retriever: ChunkRetriever,
    output_path: str,
    k: int = 15
) -> None:
    """Add retrieved chunks to exam questions and save to new file."""
    
    # Load exam
    with open(exam_path, 'r') as f:
        exam = json.load(f)
    
    # Initialize retrievers
    faiss_retriever = FAISSRetriever(chunk_retriever)
    bm25_retriever = BM25Retriever([chunk.content for chunk in chunk_retriever.chunks])
    hybrid_retriever = HybridRetriever([
        (faiss_retriever, 0.5),
        (bm25_retriever, 0.5)
    ])
    rerank_retriever = RerankingRetriever(hybrid_retriever)
    
    # Process each question
    for question in tqdm(exam):
        query = question['question']
        
        # Add retrieved chunks
        question['retrieved_chunks'] = {
            'dense': [
                {'content': str(content), 'score': float(score)}
                for content, score in faiss_retriever.retrieve(query, k=k)
            ],
            'sparse': [
                {'content': str(content), 'score': float(score)}
                for content, score in bm25_retriever.retrieve(query, k=k)
            ],
            'hybrid': [
                {'content': str(content), 'score': float(score)}
                for content, score in hybrid_retriever.retrieve(query, k=k)
            ],
            'Rerank': [
                {'content': str(content), 'score': float(score)}
                for content, score in rerank_retriever.retrieve(query, k=k)
            ]
        }
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # Save updated exam
    with open(output_path, 'w') as f:
        json.dump(exam, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    # task_domains = ["gov_report"]
    exam_files = [
        "llama_3_2_3b_single_hop_exam_processed.json",
        "gemma2_9b_single_hop_exam_processed.json",
        "ministral_8b_single_hop_exam_processed.json",
        "exam_new_llama_3_2_3b_processed_v2.json",
        "exam_new_gemma2_9b_processed_v2.json",
        "exam_new_ministral_8b_processed_v2.json",
        # "exam_new_llama_3_2_3b_processed_v2_unfiltered.json",
        # "exam_new_gemma2_9b_processed_v2_unfiltered.json"
        ]
    
    for task_domain in task_domains:
        for exam_file in exam_files:
            print(f"Starting {task_domain} - {exam_file}")
            database_dir = f"MultiHopData/{task_domain}/chunk_database"
            exam_path = f"MultiHopData/{task_domain}/exams/{exam_file}"
            # output_path = "path/to/exam_with_retrievals.json"
            
            # Load chunk retriever from saved database
            chunk_retriever = ChunkRetriever.load_database(
                database_dir,
                task_domain,
                model_name="BAAI/bge-large-en-v1.5"
            )
            
            # Add retrieved chunks to exam
            add_retrieved_chunks_to_exam(
                exam_path=exam_path,
                chunk_retriever=chunk_retriever,
                output_path=exam_path
            )
