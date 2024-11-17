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

from MultiHopData.retriever import Chunk, ChunkRetriever, HybridChunkRetriever
from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.llama.llama_instant import ModelFactory, ModelType


class ChunkAnalyser:
    def __init__(self):
        """Initialise with Mistral 7B for chunk analysis."""
        self.llm = ModelFactory.create_model(ModelType.MISTRAL_7B)
        
    def analyse_chunk_relationships(self, chunks: List[Dict[str, str]]) -> Dict[str, bool]:
        """Analyse relationships between chunks using Mistral 7B."""
        chunks_text = "\n\n".join([chunk["text"] for chunk in chunks])
        
        prompt = f"""
        Analyse the relationships between the following text chunks and identify the presence of these relationship types:
        1. Temporal sequence (events or concepts that follow a time order)
        2. Cause and effect (one concept leads to or influences another)
        3. Compare and contrast (similarities and differences between concepts)
        4. Prerequisite knowledge (one concept is required to understand another)
        5. Other

        Text chunks:
        {chunks_text}

        Provide your analysis in this format:
        {{"temporal_sequence": true/false,
        "cause_effect": true/false,
        "compare_contrast": true/false,
        "prerequisite_knowledge": true/false,
        "other": true/false}}

        Only respond with the JSON object, no other text.
        """
        
        response = self.llm.invoke(prompt)
        
        try:
            # Extract the JSON string and ensure it ends with }
            json_str = response.strip() + "}"
            extract_index = json_str.find("}\n")
            json_str = json_str[:extract_index + 1]
            return json.loads(json_str)
        except:
            # Fallback to default values if parsing fails
            return {
                "temporal_sequence": False,
                "cause_effect": False,
                "compare_contrast": False,
                "prerequisite_knowledge": False,
                "other": False
            }
    
    def identify_reasoning_type(self, chunks: List[Dict[str, str]]) -> str:
        """Identify the most appropriate reasoning type using Mistral 7B."""
        chunks_text = "\n\n".join([chunk["text"] for chunk in chunks])
        
        prompt = f"""Analyse the following text chunks and identify the most appropriate reasoning type required to connect and understand them. Choose from:
        1. bridging_inference (requiring connections between implicit information)
        2. multi_constraint_satisfaction (meeting multiple conditions)
        3. temporal_reasoning (understanding time-based relationships)
        4. causal_reasoning (understanding cause and effect)
        5. comparative_analysis (comparing and contrasting concepts)
        6. other

        Text chunks:
        {chunks_text}

        Respond with only one of the five reasoning types listed above, no other text.
        """
        
        response = self.llm.invoke(prompt)
        
        reasoning_type = response.strip()
        valid_types = [
            "bridging_inference",
            "multi_constraint_satisfaction",
            "temporal_reasoning",
            "causal_reasoning",
            "comparative_analysis",
            "other",
        ]
        
        return reasoning_type if reasoning_type in valid_types else random.choice(valid_types)


class MCQGenerator:
    def __init__(self, use_mixtral_22b: bool = False):
        """Initialise with either Mixtral 8x22B or 8x7B based on preference."""
        self.model_type = ModelType.MIXTRAL_8_22B if use_mixtral_22b else ModelType.MIXTRAL_8_7B
        self.llm = ModelFactory.create_model(self.model_type)
        self.chunk_analyser = ChunkAnalyser()
        
    def _verify_format(self, response: str) -> Tuple[bool, Dict[str, Any]]:
        """Verify the format of generated question and parse it."""
        try:
            # Extract question
            question_match = re.search(r"Question: (.*?)(?=A\))", response, re.DOTALL)
            if not question_match:
                return False, {}
            
            # Extract choices
            choices = re.findall(r"([A-D]\).*?)(?=[A-D]\)|Correct Answer|$)", response, re.DOTALL)
            if len(choices) != 4:
                return False, {}
            
            # Extract correct answer and explanation
            correct_answer_match = re.search(r"Correct Answer: ([A-D])", response)
            explanation_match = re.search(r"Explanation: (.*?)(?=Reasoning Steps|$)", response, re.DOTALL)
            reasoning_steps_match = re.search(r"Reasoning Steps: (.*?)$", response, re.DOTALL)
            
            if not all([correct_answer_match, explanation_match]):
                return False, {}
                
            parsed_response = {
                "question": question_match.group(1).strip(),
                "choices": [choice.strip() for choice in choices],
                "correct_answer": correct_answer_match.group(1),
                "explanation": explanation_match.group(1).strip(),
                "reasoning_steps": reasoning_steps_match.group(1).strip() if reasoning_steps_match else ""
            }
            
            return True, parsed_response
            
        except Exception as e:
            logging.error(f"Error parsing question format: {e}")
            return False, {}

    def generate_question(self, chunks: List[Dict[str, str]], task_domain: str) -> Optional[Dict]:
        """Generate a verified multiple-choice question."""
        # Analyse chunks
        relationships = self.chunk_analyser.analyse_chunk_relationships(chunks)
        reasoning_type = self.chunk_analyser.identify_reasoning_type(chunks)
        
        # Generate question with enhanced prompt
        prompt = make_enhanced_question_prompt(
            task_domain=task_domain,
            chunks=chunks,
            reasoning_type=reasoning_type,
            relationships=relationships
        )
        
        response = self.llm.invoke(prompt)
        
        # Verify and parse response
        is_valid, parsed_question = self._verify_format(response)
        
        if not is_valid:
            logging.warning("Generated question failed format verification")
            return None
            
        # Add metadata about generation
        parsed_question["metadata"] = {
            "reasoning_type": reasoning_type,
            "relationships": relationships,
            "num_chunks_used": len(chunks)
        }
        
        return parsed_question


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


def generate_exam(
    data: List[Dict[str, str]],
    task_domain: str,
    retriever: ChunkRetriever,
    use_mixtral_22b: bool = False
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
    use_mixtral_22b: bool = False
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
        retriever,
        use_mixtral_22b
    )

    # Save the exam to a JSON file
    with open(output_path, "w") as f:
        json.dump(exam, f, indent=2)


if __name__ == "__main__":
    task_domain = "gov_report"
    data_path = f"MultiHopData/{task_domain}/chunks/docs_chunk_semantic.json"
    output_path = f"MultiHopData/{task_domain}/exams/exam_new.json"
    sample_size = 10
    use_mixtral_22b = False  # Set to True if you want to use 22B model

    main(
        data_path,
        output_path,
        task_domain,
        sample_size,
        use_mixtral_22b=use_mixtral_22b
    )
