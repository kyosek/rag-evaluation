import json
import re
import numpy as np
import faiss
import random
import os
import logging
from typing import List, Dict, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import pickle
from tqdm import tqdm
from llama_cpp import Llama

from concurrent.futures import ThreadPoolExecutor
import threading

from MultiHopData.retriever import Chunk, ChunkRetriever, HybridChunkRetriever
from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.llama.llama_instant import CppModel




class LLMManager:
    _instances = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, model_type: ModelType, model_path: str) -> Llama:
        """Get or create a singleton instance of LlamaCpp model."""
        with cls._lock:
            if model_type not in cls._instances:
                cls._instances[model_type] = Llama(
                    model_path=os.path.join(model_path, model_type.value),
                    n_ctx=8192,  # Adjust based on your needs
                    n_threads=8   # Adjust based on your hardware
                )
            return cls._instances[model_type]

class ChunkAnalyser:
    def __init__(self, model_path: str):
        """Initialise with Mistral 7B for chunk analysis."""
        self.llm = LLMManager.get_instance(ModelType.MISTRAL_7B, model_path)
        
    def analyse_chunk_relationships(self, chunks: List[Dict[str, str]]) -> Dict[str, bool]:
        """Analyse relationships between chunks using Mistral 7B."""
        chunks_text = "\n\n".join([chunk["text"] for chunk in chunks])
        
        prompt = f"""Analyse the relationships between the following text chunks and identify the presence of these relationship types:
        1. Temporal sequence (events or concepts that follow a time order)
        2. Cause and effect (one concept leads to or influences another)
        3. Compare and contrast (similarities and differences between concepts)
        4. Prerequisite knowledge (one concept is required to understand another)

        Text chunks:
        {chunks_text}

        Provide your analysis in this format:
        {{"temporal_sequence": true/false,
        "cause_effect": true/false,
        "compare_contrast": true/false,
        "prerequisite_knowledge": true/false}}

        Only respond with the JSON object, no other text.
        """
        
        response = self.llm(
            prompt,
            max_tokens=100,
            temperature=0,
            stop=["}"]
        )
        
        try:
            # Extract the JSON string and ensure it ends with }
            json_str = response['choices'][0]['text'].strip() + "}"
            return json.loads(json_str)
        except:
            # Fallback to default values if parsing fails
            return {
                "temporal_sequence": False,
                "cause_effect": False,
                "compare_contrast": False,
                "prerequisite_knowledge": False
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

        Text chunks:
        {chunks_text}

        Respond with only one of the five reasoning types listed above, no other text.
        """
        
        response = self.llm(
            prompt,
            max_tokens=20,
            temperature=0
        )
        
        reasoning_type = response['choices'][0]['text'].strip()
        valid_types = [
            "bridging_inference",
            "multi_constraint_satisfaction",
            "temporal_reasoning",
            "causal_reasoning",
            "comparative_analysis"
        ]
        
        return reasoning_type if reasoning_type in valid_types else random.choice(valid_types)

class MCQGenerator:
    def __init__(self, model_path: str, use_mixtral_22b: bool = False):
        """Initialise with either Mixtral 8x22B or 8x7B based on preference."""
        self.model_type = ModelType.MIXTRAL_8_22B if use_mixtral_22b else ModelType.MIXTRAL_8_7B
        self.llm = LLMManager.get_instance(self.model_type, model_path)
        self.chunk_analyser = ChunkAnalyser(model_path)
        
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
        
        response = self.llm(
            prompt,
            max_tokens=1000,
            temperature=0.0
        )
        
        # Verify and parse response
        is_valid, parsed_question = self._verify_format(response['choices'][0]['text'])
        
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

def generate_exam(
    data: List[Dict[str, str]],
    step_size: int,
    task_domain: str,
    retriever: ChunkRetriever,
    model_path: str,
    use_mixtral_22b: bool = False
) -> List[Dict[str, str]]:
    """
    Generate an exam with multiple-choice questions from the given data.
    """
    mcq_generator = MCQGenerator(model_path, use_mixtral_22b)
    exam = []

    for k in tqdm(range(0, len(data), step_size)):
        # Get the current chunk and its similar chunks
        current_chunk = data[k]
        chunk_data = Chunk(
            chunk_id=current_chunk["chunk_id"],
            doc_id=current_chunk["doc_id"],
            content=current_chunk["content"],
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
                "chunk_id": current_chunk["chunk_id"],
                "doc_id": current_chunk["doc_id"],
                "text": current_chunk["content"],
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
    model_path: str,
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
        step_size,
        task_domain,
        retriever,
        model_path,
        use_mixtral_22b
    )

    # Save the exam to a JSON file
    with open(output_path, "w") as f:
        json.dump(exam, f, indent=2)

if __name__ == "__main__":
    task_domain = "SecFilings"
    data_path = f"MultiHopData/{task_domain}/chunks/docs_chunk_semantic.json"
    output_path = f"MultiHopData/{task_domain}/exams/exam_new.json"
    model_path = "path/to/your/models"  # Update with your model path
    sample_size = 1500
    use_mixtral_22b = False  # Set to True if you want to use 22B model

    main(
        data_path,
        output_path,
        task_domain,
        sample_size,
        model_path=model_path,
        use_mixtral_22b=use_mixtral_22b
    )
