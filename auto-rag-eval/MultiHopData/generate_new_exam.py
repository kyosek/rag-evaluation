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

from MultiHopData.prompt_template import PromptTemplate
from MultiHopData.retriever import Chunk, ChunkRetriever, HybridChunkRetriever
from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.llama.llama_instant import ModelFactory, ModelType


class ChunkAnalyser:
    def __init__(self):
        """Initialise with Mistral 7B for chunk analysis."""
        # self.llm = ModelFactory.create_model(ModelType.MISTRAL_7B)
        self.llm = ModelFactory.create_model(ModelType.LLAMA_3_2_3B)
        # self.llm = ModelFactory.create_model(ModelType.PHI_2)
        
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
    def __init__(self, model_name: str = None):
        """
        Initialise the MCQ Generator with a specific model.
        
        Args:
            model_name (str, optional): Name of the model to use.
            If None, uses a default model.
        """
        # Mapping of model names to ModelType enums
        self.model_mapping = {
            'llama_3_1_8b': ModelType.LLAMA_3_1_8B,
            'llama_3_2_3b': ModelType.LLAMA_3_2_3B,
            'mistral_7b': ModelType.MISTRAL_7B,
        }
        
        # Select model based on input or use default
        if model_name:
            # Convert to lowercase to handle case-insensitive input
            model_name = model_name.lower()
            
            if model_name not in self.model_mapping:
                raise ValueError(f"Unsupported model: {model_name}. "
                                 f"Supported models: {list(self.model_mapping.keys())}")
            
            self.model_type = self.model_mapping[model_name]
        else:
            # Default model if no name provided
            self.model_type = ModelType.LLAMA_3_2_3B
        
        print(f"Using {self.model_type}")
        self.llm = ModelFactory.create_model(self.model_type)
        # self.chunk_analyser = ChunkAnalyser()
    
    def extract_with_patterns(self, text: str, patterns: List) -> List[str]:

        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    return matches
            except re.error:
                continue
        return None

    def _extract_question(self, response: str) -> Optional[str]:
        """Extract question from response with improved pattern matching."""
        question_patterns = [
            r"Question:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"Question 1:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"question:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"question 1:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"documentation:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",  # for ClaudeV2 mostly
            r"### Assistant: (.*?)\n",
        ]
        # Extract the question
        question_matches = self.extract_with_patterns(response, question_patterns)
        question = question_matches[0].strip() if question_matches else None
        return question

    def _extract_choices(self, response: str) -> Optional[List[str]]:
        """Extract and validate choices with robust pattern matching."""
        # Match choices including possible line breaks but excluding explanation sections
        choices_patterns = [
            r"([A-D]\) .*?)(?=$|\n[A-D]\)|\n\n)",
            r"([A-D]\)(?:.|\n)*?)(?=$|\n[A-D]\)|\n\n)",
            r"([A-D]\. .*?)(?=$|\n[A-D]\.|\n\n)",
            r"([A-D]\.)(?:.|\n)*?)(?=$|\n[A-D]\.|\n\n)",
            r"([1-4]\) .*?)(?=$|\n[1-4]\)|\n\n)",
            r"([1-4]\)(?:.|\n)*?)(?=$|\n[1-4]\)|\n\n)",
            r"([1-4]\. .*?)(?=$|\n[1-4]\.|\n\n)",
            r"([1-4]\.)(?:.|\n)*?)(?=$|\n[1-4]\.|\n\n)",
            r"([a-d]\) .*?)(?=$|\n[a-d]\)|\n\n)",
            r"([a-d]\)(?:.|\n)*?)(?=$|\n[a-d]\)|\n\n)",
            r"([a-d]\. .*?)(?=$|\n[a-d]\.|\n\n)",
            r"([a-d]\.)(?:.|\n)*?)(?=$|\n[a-d]\.|\n\n)",
        ]
        choices_matches = self.extract_with_patterns(response, choices_patterns)
        choices = [match.strip() for match in choices_matches] if choices_matches else None
        
        # Only keep first 4 answers
        choices = (
            choices[:4]
            if choices
            and len(choices) >= 4
            and len(set([choice[0] for choice in choices[:4]])) == 4
            else None
        )
        
        # Remove scenarios with empty answers ['A)], 'B)', 'C)', 'D)']
        choices = choices if choices and min([len(choice) for choice in choices]) > 2 else None
        return choices

    def _extract_correct_answer(self, response: str) -> Optional[str]:
        """Extract correct answer with validation."""
        # Try first with full pattern including 'Correct Answer:'
        correct_answer_match = re.search(r"Correct Answer:\s*([A-D])\)?", response, re.IGNORECASE)
        
        # If first method fails, try a more lenient approach
        if not correct_answer_match:
            correct_answer_match = re.search(r"\*\*Correct Answer:\*\*\s*([A-D])\)?", response, re.IGNORECASE)
        
        # If still no match, try finding a lone capital letter at the end
        if not correct_answer_match:
            correct_answer_match = re.search(r"([A-D])$", response.split('\n')[-1].strip(), re.IGNORECASE)
        
        # Return the correct answer if found
        if correct_answer_match:
            return f"{correct_answer_match.group(1)})"
        
        return None

    def _extract_reasoning_steps(self, response: str) -> Optional[str]:
        """Extract reasoning steps if available."""
        reasoning_match = re.search(r"Reasoning Steps:\s*(.*?)(?=(?:\n\s*[A-D]\)|\Z))", response, re.DOTALL)
        return reasoning_match.group(1).strip() if reasoning_match else None
    
    def _make_enhanced_question_prompt(self, task_domain: str, chunks: List[Dict[str, str]]) -> str:
        """Create a prompt using the appropriate template for the specified model."""
        documentation = "\n\n".join([f"Chunk{i}: {chunk['text']}" for i, chunk in enumerate(chunks)])
        
        template = PromptTemplate.get_question_generation_prompt_template(self.model_type, chunks, task_domain, documentation)
        return template
        
    def generate_question(self, chunks: List[Dict[str, str]], task_domain: str) -> Optional[Dict]:
        """Generate a multiple-choice question with documentation included."""
        # Analyse chunks
        # relationships = self.chunk_analyser.analyse_chunk_relationships(chunks)
        # reasoning_type = self.chunk_analyser.identify_reasoning_type(chunks)
        
        # Generate question with enhanced prompt
        prompt = self._make_enhanced_question_prompt(
            task_domain=task_domain,
            chunks=chunks,
            # reasoning_type=reasoning_type,
            # relationships=relationships
        )
        
        response = self.llm.invoke(prompt)
        
        try:
            # Create the question dictionary with documentation
            parsed_question = {
                "question": self._extract_question(response),
                "choices": self._extract_choices(response),
                "correct_answer": self._extract_correct_answer(response),
                "documentation": [chunk["text"] for chunk in chunks],
                "metadata": {
                    # "reasoning_type": reasoning_type,
                    # "relationships": relationships,
                    "num_chunks_used": len(chunks)
                }
            }
            
            # Add reasoning steps if available
            # if reasoning_steps_match:
            #     parsed_question["metadata"]["reasoning_steps"] = self._extract_correct_answer(response)
            
            return parsed_question
            
        except Exception as e:
            logging.error(f"Error parsing question format: {e}")
            return None


def generate_exam(
    data: List[Dict[str, str]],
    task_domain: str,
    model_name: str,
    retriever: ChunkRetriever,
    target_hop_number: int = 250,
) -> List[Dict[str, str]]:
    """
    Generate an exam with multiple-choice questions from the given data.
    """
    mcq_generator = MCQGenerator(model_name)
    exam = []
    hop_counts = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
    }

    for k in tqdm(range(0, len(data))):
        # Get the current chunk and its similar chunks
        current_chunk = data[k]
        chunk_data = Chunk(
            chunk_id=current_chunk.chunk_id,
            doc_id=current_chunk.doc_id,
            content=current_chunk.content,
            original_index=k,
        )
        
        hop_try_count = 0
        while True:
            num_hops = random.randint(1, 4)
            if hop_counts[str(num_hops)] < target_hop_number:
                break
            hop_try_count += 1
            if hop_try_count >= 3:
                break
        
        similar_chunks = retriever.find_similar_chunks(
            chunk_data, k=num_hops, similarity_threshold=0.01, exclude_same_doc=False
        )
        
        hop_counts[str(len(similar_chunks))] += 1

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
    model_name: str,
    task_domain: str,
    sample_size: int,
    target_hop_number: int = 250
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
        model_name,
        retriever,
        target_hop_number
    )

    # Save the exam to a JSON file
    with open(output_path, "w") as f:
        json.dump(exam, f, indent=2)


if __name__ == "__main__":
    sample_size = 1200
    target_hop_number = 301
    
    assert sample_size < target_hop_number * 4
    
    # task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    task_domains = ["gov_report", "multifieldqa_en", "SecFilings"]
    
    # model_names = ['llama_3_2_3b', 'llama_3_1_8b']
    model_names = ['llama_3_1_8b']
    
    # task_domain = "gov_report"
    for model_name in model_names:
        for task_domain in task_domains:
            data_path = f"MultiHopData/{task_domain}/chunks/docs_chunk_semantic_cleaned.json"
            output_path = f"MultiHopData/{task_domain}/exams/exam_new_{model_name}.json"

            main(
                data_path,
                output_path,
                model_name,
                task_domain,
                sample_size,
                target_hop_number=target_hop_number
            )
