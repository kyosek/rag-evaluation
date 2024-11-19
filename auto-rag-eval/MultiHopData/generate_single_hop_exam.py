import json
import re
import numpy as np
import faiss
import random
import os
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import pickle
from tqdm import tqdm
from llama_cpp import Llama

from concurrent.futures import ThreadPoolExecutor
import threading

from MultiHopData.generate_new_exam import MCQGenerator
from MultiHopData.prompt_template import PromptTemplate
from MultiHopData.retriever import Chunk, ChunkRetriever, HybridChunkRetriever
from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.llama.llama_instant import ModelFactory, ModelType


def generate_exam(
    data: List[Dict[str, str]],
    task_domain: str,
    use_mixtral_22b: bool = False,
) -> List[Dict[str, str]]:
    """
    Generate an exam with multiple-choice questions from the given data.
    """
    mcq_generator = MCQGenerator(use_mixtral_22b)
    exam = []

    for k in tqdm(range(0, len(data))):
        # Get the current chunk and its similar chunks
        current_chunk = data[k]
        chunk_data = Chunk(
            chunk_id=current_chunk.chunk_id,
            doc_id=current_chunk.doc_id,
            content=current_chunk.content,
            original_index=k,
        )
        
        chunk_dict = [
            {
                "chunk_id": current_chunk.chunk_id,
                "doc_id": current_chunk.doc_id,
                "text": current_chunk.content,
            }
        ]
        
        try:
            question_data = mcq_generator.generate_question(chunk_dict, task_domain)
            if question_data:
                exam.append(question_data)
        except Exception as e:
            logging.error(f"Error generating question: {e}")
            continue

    return exam


def main(
    data_path: str,
    output_path: str,
    task_domain: str,
    sample_size: int,
    use_mixtral_22b: bool = False,
):
    logging.info("Start processing")
    retriever = HybridChunkRetriever(task_domain, random_seed=42)

    if not os.path.exists(f"MultiHopData/{task_domain}/chunk_database"):
        logging.info("Load documents")
        retriever.load_documents(data_path)
        logging.info(f"Save the database to 'MultiHopData/{task_domain}/chunk_database'")
        retriever.save_database(f"MultiHopData/{task_domain}/chunk_database")
    else:
        logging.info("Loading database from file")
        retriever = HybridChunkRetriever.load_database(
            f"MultiHopData/{task_domain}/chunk_database", task_domain
        )

    # Sample chunks with a specific seed
    sampled_chunks = retriever.sample_chunks(sample_size, seed=42)

    # Generate the exam
    print("Start generating exam")
    exam = generate_exam(
        sampled_chunks,
        task_domain,
        use_mixtral_22b
    )

    # Save the exam to a JSON file
    with open(output_path, "w") as f:
        json.dump(exam, f, indent=2)


if __name__ == "__main__":
    sample_size = 1200
    use_mixtral_22b = False  # Set to True if you want to use 22B model
    
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    
    # task_domain = "gov_report"
    for task_domain in task_domains:
        data_path = f"MultiHopData/{task_domain}/chunks/docs_chunk_semantic_cleaned.json"
        output_path = f"MultiHopData/{task_domain}/exams/exam_new.json"

        main(
            data_path,
            output_path,
            task_domain,
            sample_size,
            use_mixtral_22b=use_mixtral_22b,
        )
