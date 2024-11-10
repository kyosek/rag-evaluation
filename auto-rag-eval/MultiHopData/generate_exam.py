import json
import re
import numpy as np
import faiss
import random
import os
import logging
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import pickle
from tqdm import tqdm

from MultiHopData.retriever import Chunk, ChunkRetriever
from LLMServer.gcp.claude_instant import ClaudeGcp


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
        chunk_data = Chunk(
                chunk_id=current_chunk.chunk_id,
                doc_id=current_chunk.doc_id,
                content=current_chunk.content,
                original_index=k
            )
        similar_chunks = retriever.find_similar_chunks(
            chunk_data,
            k=4,
            similarity_threshold=0.9,
            exclude_same_doc=False
        )
        
        if not similar_chunks:
            similar_chunks = retriever.find_similar_chunks(
                chunk_data,
                k=1,
                similarity_threshold=0.01,
                exclude_same_doc=False
        )
        
        chunk_dict = [{"chunk_id": current_chunk.chunk_id, "doc_id": current_chunk.doc_id, "text": current_chunk.content}]
        chunk_dict += [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.content} for c, _ in similar_chunks]
        
        try:
            # Generate a high-level (L3) question
            question_prompt = make_l3_question_prompt(task_domain, chunk_dict)
            answer = llm_model.invoke(prompt=question_prompt, params={})
            
            # Extract question, choices, correct answer, and explanation
            question = answer.split("\nQuestion: ")[1].split("\nA)")[0]
            correct_answer = answer.split("\nCorrect Answer: ")[1].split("\nExplanation:")[0]
            explanation = answer.split("\nExplanation: ")[1]
            # Find all choices that start with A), B), C), or D)
            choices = re.findall(r'[A-D]\)(.*?)(?=[A-D]\)|Correct Answer|$)', answer, re.DOTALL)

            # Clean up the choices by removing extra whitespace and newlines
            choices = [f"{chr(65+i)}) {choice.strip()}" for i, choice in enumerate(choices)]
            
            # Construct the exam entry
            exam_entry = {
                "question": question,
                "documentation": [chunk["text"] for chunk in chunk_dict],
                "choices": [choice.strip() for choice in choices],
                "correct_answer": f"{correct_answer}) {explanation}"
            }
            
            exam.append(exam_entry)
        except:
            pass
    
    return exam


def make_l3_question_prompt(task_domain: str, chunks: List[Dict[str, str]]) -> str:
    documentation = "\n\n".join([f"Chunk{i}: {chunk['text']}" for i, chunk in enumerate(chunks)])
    return f"""
    <<SYS>>
    You are an expert exam question and answer generator specializing in creating high-quality, challenging multiple-choice questions. 
    Your questions should:
    1. Have 4 choices of A), B), C), D) - 1 correct and 3 incorrect choices
    2. Target L3 (Analysis/Application) or higher cognitive levels in Bloom's taxonomy
    3. Require integration of multiple concepts from the documentation
    4. Test critical thinking rather than memorization
    5. Have carefully crafted 3 distractors that represent common misconceptions

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


def main(data_path: str, output_path: str, task_domain: str, sample_size: int, step_size: int):
    logging.info("Start processing")
    retriever = ChunkRetriever(task_domain, random_seed=42)
    model = ClaudeGcp()

    if not os.path.exists(f'MultiHopData/{task_domain}/chunk_database'):
        logging.info("Load documents")
        retriever.load_documents(data_path)
        
        logging.info(f"Save the database to 'MultiHopData/{task_domain}/chunk_database'")
        retriever.save_database(f'MultiHopData/{task_domain}/chunk_database')
    else:
        logging.info("Loading database from file")
        retriever = ChunkRetriever.load_database(f'MultiHopData/{task_domain}/chunk_database')
    
    # Sample chunks with a specific seed
    sampled_chunks = retriever.sample_chunks(sample_size, seed=42)
    
    # Generate the exam
    print("Start generating exam")
    exam = generate_exam(sampled_chunks, step_size, task_domain, retriever, model)
    
    # Save the exam to a JSON file
    with open(output_path, "w") as f:
        json.dump(exam, f, indent=2)


if __name__ == "__main__":
    task_domain = "wiki"
    data_path = f"MultiHopData/{task_domain}/docs_chunk.json"
    output_path = f"MultiHopData/{task_domain}/exam.json"
    sample_size = 1100
    
    main(data_path, output_path, task_domain, sample_size, step_size=1)
