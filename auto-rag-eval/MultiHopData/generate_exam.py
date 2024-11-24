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

from MultiHopData.retriever import Chunk, ChunkRetriever, HybridChunkRetriever
from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.llama.llama_instant import CppModel


class MCQGenerator:
    def __init__(self, llm_model):
        self.llm = llm_model
        
    def generate_question(self, chunks: List[Dict[str, str]], task_domain: str) -> Dict:
        """Generate a difficult multiple-choice question using multiple chunks."""
        
        # 1. Analyze relationships between chunks
        chunk_relationships = self._analyze_chunk_relationships(chunks)
        
        # 2. Identify potential reasoning types
        reasoning_type = self._identify_reasoning_type(chunks)
        
        # 3. Generate question with specific reasoning type
        question_prompt = self._create_targeted_prompt(
            chunks, 
            task_domain,
            reasoning_type,
            chunk_relationships
        )
        
        # 4. Generate initial response
        response = self.llm.invoke(prompt=question_prompt, params={})
        
        # 5. Validate and enhance distractors
        validated_response = self._validate_and_enhance(response, chunks)
        
        # 6. Verify format
        if not self._verify_format(validated_response):
            raise ValueError("Generated question format is invalid")
            
        return validated_response
    
    def _analyze_chunk_relationships(self, chunks: List[Dict[str, str]]) -> Dict:
        """Analyze how chunks relate to each other."""
        relationships = {
            "temporal_sequence": False,
            "cause_effect": False,
            "compare_contrast": False,
            "prerequisite_knowledge": False
        }
        
        # Add logic to detect relationships between chunks
        return relationships
        
    def _identify_reasoning_type(self, chunks: List[Dict[str, str]]) -> str:
        """Identify potential reasoning types based on chunk content."""
        reasoning_types = [
            "bridging_inference",
            "multi_constraint_satisfaction",
            "temporal_reasoning",
            "causal_reasoning",
            "comparative_analysis"
        ]
        # Add logic to select appropriate reasoning type
        return random.choice(reasoning_types)
    
    def _verify_format(self, response: Dict) -> bool:
        """Verify the format of generated question."""
        required_fields = ['question', 'choices', 'correct_answer', 'explanation']
        if not all(field in response for field in required_fields):
            return False
            
        # Verify choices format
        if len(response['choices']) != 4:
            return False
            
        # Verify each choice starts with A), B), C), or D)
        choice_pattern = re.compile(r'^[A-D]\)')
        if not all(choice_pattern.match(choice) for choice in response['choices']):
            return False
            
        # Verify correct answer is one of A, B, C, or D
        if not response['correct_answer'][0] in ['A', 'B', 'C', 'D']:
            return False
            
        return True


def generate_exam(
    data: List[Dict[str, str]],
    step_size: int,
    task_domain: str,
    retriever: ChunkRetriever,
    llm_model,
) -> List[Dict[str, str]]:
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
            original_index=k,
        )
        similar_chunks = retriever.find_similar_chunks(
            chunk_data, k=4, similarity_threshold=0.01, exclude_same_doc=False
        )

        if not similar_chunks:
            similar_chunks = retriever.find_similar_chunks(
                chunk_data, k=1, similarity_threshold=0.01, exclude_same_doc=False
            )

        chunk_dict = [
            {
                "chunk_id": current_chunk.chunk_id,
                "doc_id": current_chunk.doc_id,
                "text": current_chunk.content,
            }
        ]
        chunk_dict += [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.content}
            for c, _ in similar_chunks
        ]

        try:
            # question_prompt = make_l3_question_prompt(task_domain, chunk_dict)
            question_prompt = make_enhanced_question_prompt(task_domain, chunk_dict)
            answer = llm_model.invoke(prompt=question_prompt, params={})

            # Extract question, choices, correct answer, and explanation
            question = answer.split("\nQuestion: ")[1].split("\nA)")[0]
            correct_answer = answer.split("\nCorrect Answer: ")[1].split("\nExplanation:")[0]
            explanation = answer.split("\nExplanation: ")[1]
            # Find all choices that start with A), B), C), or D)
            choices = re.findall(r"[A-D]\)(.*?)(?=[A-D]\)|Correct Answer|$)", answer, re.DOTALL)

            # Clean up the choices by removing extra whitespace and newlines
            choices = [f"{chr(65+i)}) {choice.strip()}" for i, choice in enumerate(choices)]

            # Construct the exam entry
            exam_entry = {
                "question": question,
                "documentation": [chunk["text"] for chunk in chunk_dict],
                "choices": [choice.strip() for choice in choices],
                "correct_answer": f"{correct_answer}) {explanation}",
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


def make_enhanced_question_prompt(
    task_domain: str,
    chunks: List[Dict[str, str]],
    reasoning_type: str,
    relationships: Dict[str, bool]
) -> str:
    documentation = "\n\n".join([f"Chunk{i}: {chunk['text']}" for i, chunk in enumerate(chunks)])
    
    return f"""
    <<SYS>>
    You are an expert exam question generator specializing in creating challenging multiple-choice questions that require complex reasoning across multiple pieces of information.
    
    Required reasoning type: {reasoning_type}
    Identified relationships between chunks: {relationships}
    
    Core requirements:
    1. Question MUST require synthesizing information from at least {len(chunks)} different chunks
    2. Distractors must be highly plausible and based on common misconceptions or partial understanding
    3. The correct answer should not be obvious without carefully analyzing all chunks
    4. Each distractor should represent a different type of reasoning error
    
    Question Design Principles:
    1. Incorporate subtle dependencies between chunks
    2. Require careful analysis of conditional statements
    3. Include scenarios where surface-level reading might lead to wrong conclusions
    4. Design distractors that would be chosen if key information from certain chunks is missed
    
    Format Requirements:
    - Question text should be clear but complex
    - Each option must start with A), B), C), or D)
    - Include detailed explanation of why each distractor is incorrect
    <</SYS>>

    Domain: {task_domain}
    Documentation: {documentation}

    Generate a question following this format:
    Question: [Complex question requiring multi-hop reasoning]
    A) [Option incorporating some but not all key information]
    B) [Option based on common misinterpretation]
    C) [Option that would be correct if one crucial detail is missed]
    D) [Correct option requiring synthesis of all chunks]
    Correct Answer: [Letter]
    Explanation: [Detailed explanation of why the answer is correct AND why each distractor is incorrect]
    Reasoning Steps: [Step-by-step breakdown of how to arrive at the correct answer]
    """


def main(data_path: str, output_path: str, task_domain: str, sample_size: int, model_name: str):
    logging.info("Start processing")
    retriever = HybridChunkRetriever(task_domain, random_seed=42)
    model = CppModel(model_name=model_name)
    exam_generator = MCQGenerator(model)

    if not os.path.exists(f"MultiHopData/{task_domain}/chunk_database"):
        logging.info("Load documents")
        retriever.load_documents(data_path)

        logging.info(f"Save the database to 'MultiHopData/{task_domain}/chunk_database'")
        retriever.save_database(f"MultiHopData/{task_domain}/chunk_database", task_domain)
    else:
        logging.info("Loading database from file")
        retriever = HybridChunkRetriever.load_database(
            f"MultiHopData/{task_domain}/chunk_database", task_domain
        )

    # Sample chunks with a specific seed
    sampled_chunks = retriever.sample_chunks(sample_size, seed=42)

    # Generate the exam
    print("Start generating exam")
    exam = generate_exam(sampled_chunks, step_size, task_domain, retriever, model)

    # Save the exam to a JSON file
    with open(output_path, "w") as f:
        json.dump(exam, f, indent=2)


if __name__ == "__main__":
    task_domain = "SecFilings"
    data_path = f"MultiHopData/{task_domain}/chunks/docs_chunk_semantic.json"
    output_path = f"MultiHopData/{task_domain}/exams/exam_new.json"
    sample_size = 1500
    model_name = ""

    main(data_path, output_path, task_domain, sample_size, model_name)
