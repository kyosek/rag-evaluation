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
            'ministral_8b': ModelType.MINISTRAL_8B,
            "gemma2_9b": ModelType.GEMMA2_9B,
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
            r"\*\*Question\*\*\n\n(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"Question:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"Question 1:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"question:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"question 1:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"documentation:(.*?)(?:\n[a-dA-D1-4]\)|\n\n[a-dA-D1-4]\))",
            r"### Assistant: (.*?)\n",
        ]
        # Extract the question
        question_matches = self.extract_with_patterns(response, question_patterns)
        question = question_matches[0].strip() if question_matches else None
        return question

    def _extract_choices(self, response: str) -> Optional[List[str]]:
        """
        Extract and validate multiple choice answers with improved multi-line handling.
        
        Args:
            response (str): The raw response text containing the MCQ
            
        Returns:
            Optional[List[str]]: List of choices if found and valid, None otherwise
        """
        # Try different patterns in order of specificity
        patterns = [
            # Basic pattern for lettered choices with parentheses
            r'(?:^|\n)([A-D]\))\s*(.*?)(?=\n[A-D]\)|$)',
            
            # Pattern for choices with newlines and content
            r'\n([A-D]\))\s*((?:(?!\n[A-D]\)).)*)',
            
            # Pattern for full answers with possible multiline content
            r'(?:^|\n)([A-D]\))\s*((?:(?!\n[A-D]\)|Correct Answer:|Distractors:).)*)',
            
            # Fallback pattern for simple format
            r'([A-D]\))\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.MULTILINE | re.DOTALL)
            choices = []
            
            for match in matches:
                identifier = match.group(1)
                content = match.group(2).strip() if len(match.groups()) > 1 else ''
                full_choice = f"{identifier} {content}"
                # Clean up any extra whitespace
                full_choice = ' '.join(full_choice.split())
                choices.append(full_choice)
            
            choices = (
                choices[:4]
                if choices
                and len(choices) >= 4
                and len(set([choice[0] for choice in choices[:4]])) == 4
                else None
            )
            # Validate we have exactly 4 choices
            if len(choices) == 4 and len(set([c[0] for c in choices])) == 4:
                return choices
        
        return None

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
    
    def _extract_single_chunk_answerable(self, response: str) -> Optional[bool]:
        """Extract single_chunk_answerable from verdict response."""
        patterns = [
            r"\"single_chunk_answerable\":\s*(true|false)",
            r"single_chunk_answerable:\s*(true|false)",
            r"Single chunk answerable:\s*(true|false)",
            r"Can be answered with single chunk:\s*(yes|no|true|false)",
        ]
        matches = self.extract_with_patterns(response, patterns)
        if matches:
            value = matches[0].lower()
            return value in ['true', 'yes']
        return None

    def _extract_required_chunks(self, response: str) -> Optional[List[int]]:
        """Extract required_chunks from verdict response."""
        patterns = [
            r"\"required_chunks\":\s*\[([\d\s,]+)\]",
            r"required_chunks:\s*\[([\d\s,]+)\]",
            r"Required chunks:\s*\[([\d\s,]+)\]",
            r"Chunks needed:\s*\[([\d\s,]+)\]",
            r"Required chunks:[\s\n]*(\d+(?:,\s*\d+)*)",
            r"Chunks needed:[\s\n]*(\d+(?:,\s*\d+)*)",
        ]
        matches = self.extract_with_patterns(response, patterns)
        if matches:
            # Clean and parse the matched string
            chunk_str = matches[0].strip('[]').replace(' ', '')
            try:
                return [int(x) for x in chunk_str.split(',') if x]
            except ValueError:
                return None
        return None

    def _extract_synthesis_required(self, response: str) -> Optional[bool]:
        """Extract synthesis_required from verdict response."""
        patterns = [
            r"\"synthesis_required\":\s*(true|false)",
            r"synthesis_required:\s*(true|false)",
            r"Synthesis required:\s*(true|false)",
            r"Requires synthesis:\s*(yes|no|true|false)",
        ]
        matches = self.extract_with_patterns(response, patterns)
        if matches:
            value = matches[0].lower()
            return value in ['true', 'yes']
        return None

    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning from verdict response."""
        patterns = [
        # Handle JSON-style with quotes
        r'\"reasoning\":\s*\"((?:[^\"\\]|\\.)*)\"',  # Matches JSON format with escaped quotes
        r'"reasoning":\s*"([^"]*)"',  # Simple JSON quoted format
        # Handle JSON-style without quotes
        r'"reasoning":\s*(.*?)(?=\s*[,}\n])',  # Unquoted JSON format
        r'reasoning":\s*(.*?)(?=\s*[,}\n])',   # Alternative unquoted format
        # Handle plain text formats
        r'reasoning:\s*(.*?)(?=\n\s*[a-z_"]+:|\n\s*\{|\n\s*\}|$)',  # Matches until next field or end
        r'Reasoning:\s*(.*?)(?=\n\s*[a-z_"]+:|\n\s*\{|\n\s*\}|$)',
        r'Explanation:\s*(.*?)(?=\n\s*[a-z_"]+:|\n\s*\{|\n\s*\}|$)',
    ]
        matches = self.extract_with_patterns(response, patterns)
        if matches:
            # Clean up the extracted reasoning
            reasoning = matches[0].strip()
            # Handle escaped quotes if present
            reasoning = reasoning.replace('\\"', '"').replace('\\\\', '\\')
            # Remove any trailing commas or syntax artifacts
            reasoning = re.sub(r'[,\s]+$', '', reasoning)
            return reasoning
        return None

    def _extract_missing_information(self, response: str) -> Optional[str]:
        """Extract missing_information from verdict response."""
        patterns = [
            r"\"missing_information\":\s*\"(.*?)\"(?=,|\})",
            r"missing_information:\s*(.*?)(?=\n|$)",
            r"Missing information:\s*(.*?)(?=\n|$)",
            r"Missing:\s*(.*?)(?=\n|$)",
        ]
        matches = self.extract_with_patterns(response, patterns)
        return matches[0].strip() if matches else None

    def _extract_confidence(self, response: str) -> Optional[int]:
        """Extract confidence score from verdict response."""
        patterns = [
            r"\"confidence\":\s*(\d+)",
            r"confidence:\s*(\d+)",
            r"Confidence:\s*(\d+)",
            r"Confidence level:\s*(\d+)",
        ]
        matches = self.extract_with_patterns(response, patterns)
        if matches:
            try:
                confidence = int(matches[0])
                return confidence if 1 <= confidence <= 5 else None
            except ValueError:
                return None
        return None

    def _extract_verdict(self, response: str) -> Dict:
        """Extract all verdict components using pattern matching."""
        verdict = {
            "single_chunk_answerable": self._extract_single_chunk_answerable(response),
            "required_chunks": self._extract_required_chunks(response),
            "synthesis_required": self._extract_synthesis_required(response),
            "reasoning": self._extract_reasoning(response),
            "missing_information": self._extract_missing_information(response),
            "confidence": self._extract_confidence(response)
        }
        
        # Validate that we have at least the critical fields
        if (verdict["required_chunks"] is not None and
            verdict["reasoning"] is not None):
            return verdict
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
                "metadata": {"num_chunks_used": len(chunks)}
            }
            
            # Add reasoning steps if available
            # if reasoning_steps_match:
            #     parsed_question["metadata"]["reasoning_steps"] = self._extract_correct_answer(response)
            
            return parsed_question
            
        except Exception as e:
            logging.error(f"Error parsing question format: {e}")
            return None

    def _regenerate_question_with_feedback(
        self,
        chunks: List[Dict[str, str]],
        task_domain: str,
        feedback: str
        ) -> Optional[Dict]:
        """Regenerate question using verification feedback."""
        enhanced_prompt = self._make_enhanced_question_prompt(
            task_domain=task_domain,
            chunks=chunks,
        ) + f"\n\nPrevious attempt feedback: {feedback}\nPlease ensure the question requires synthesising information across multiple chunks."
        
        response = self.llm.invoke(enhanced_prompt)
        
        try:
            return {
                "question": self._extract_question(response),
                "choices": self._extract_choices(response),
                "correct_answer": self._extract_correct_answer(response),
                "documentation": [chunk["text"] for chunk in chunks],
                "metadata": {
                    "num_chunks_used": len(chunks)
                }
            }
        except Exception as e:
            logging.error(f"Error parsing regenerated question: {e}")
            return None

    def call_verification_agent(self, question_data: dict, chunks: List[Dict[str, str]], 
                    task_domain: str, target_hops: int, max_attempts: int = 3) -> Dict:
        """
        Verify and potentially regenerate the question to ensure it requires the target number of hops.
        
        Args:
            question_data: The generated question data
            chunks: List of document chunks
            task_domain: Domain of the task
            target_hops: Target number of hops required
            
        Returns:
            Dict containing the final question data with verification metadata
        """
        verification_attempts = 0
        current_question = question_data
        verdicts = []
        
        while verification_attempts < max_attempts:
            # Generate verification prompt
            verification_prompt = PromptTemplate.get_verification_prompt(current_question, chunks)
            
            # Get verdict from LLM
            verdict_response = self.llm.invoke(verification_prompt)
            verdict = self._extract_verdict(verdict_response)
            if verdict:
                verdicts.append(verdict)
                
                # Check if the question meets the hop requirement
                if len(verdict['required_chunks']) == target_hops:
                    break
                
                # Regenerate question with feedback
                verification_attempts += 1
                if verification_attempts < max_attempts:
                    regenerated_question = self._regenerate_question_with_feedback(
                        chunks=chunks,
                        task_domain=task_domain,
                        feedback=verdict['reasoning']
                    )
                    if regenerated_question:
                        current_question = regenerated_question
            else:
                verification_attempts += 1
                logging.error("Failed to extract verdict from response")
        
        # Add verification metadata to the final question
        current_question['metadata'].update({
            'verification_attempts': verification_attempts,
            'verification_verdicts': verdicts,
            'final_verdict': verdicts[-1] if verdicts else None,
            'meets_hop_requirement': (len(verdicts[-1]['required_chunks']) == target_hops) if verdicts else False
        })
        
        return current_question


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
    num_questions = len(data)
    exam = []
    hop_counts = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
    }

    for ith_question in tqdm(range(0, num_questions)):
        # Get the current chunk and its similar chunks
        current_chunk = data[ith_question]
        chunk_data = Chunk(
            chunk_id=current_chunk.chunk_id,
            doc_id=current_chunk.doc_id,
            content=current_chunk.content,
            original_index=ith_question,
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
                question_data = mcq_generator.call_verification_agent(
                    question_data=question_data,
                    chunks=chunk_dict,
                    task_domain=task_domain,
                    target_hops=len(similar_chunks) + 1  # +1 because we include the original chunk
                    )
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
    print("Start generating the exam")
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
    sample_size = 700
    target_hop_number = 176
    
    assert sample_size < target_hop_number * 4
    
    # task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    task_domains = ["gov_report"]
    
    # model_names = ['llama_3_2_3b', "gemma2_9b", 'ministral_8b']
    model_names = ['llama_3_2_3b']
    
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
