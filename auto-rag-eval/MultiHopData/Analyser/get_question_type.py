import json
import random
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from LLMServer.llama.llama_instant import ModelFactory, ModelType


class ChunkAnalyser:
    def __init__(self):
        """Initialise with Mistral 7B for chunk analysis."""
        self.llm = ModelFactory.create_model(ModelType.GEMMA2_9B)
    
    def analyse_question_type(self, question_data: dict) -> Dict[str, bool]:
        """Analyse question type"""
        options = [choice.split(') ')[1] for choice in question_data['choices']]
        
        chunk_dict = [
            {
                "text": question_data["documentation"],
            }
        ]
        chunk_text = '\n'.join([f"Chunk{i+1}: {chunk['text']}" for i, chunk in enumerate(chunk_dict)])
        
        question=question_data['question']
        option_a=options[0]
        option_b=options[1]
        option_c=options[2]
        option_d=options[3]
        correct_answer=question_data['correct_answer']
        
        prompt = f"""<start_of_turn>user
        You are an AI assistant to analyse an exam question type.
        You will be given a multiple-choice question and supporting documents.
        Based on those information, you classify the question into one of the following question type:

        Question types:
        - Inference: the correct answer must be deduced through reasoning from the given supporting documents
        - Comparison: the correct answer requires to compare multiple supporting documents' information
        - Temporal: the correct answer requires a reasoning of events or concepts that follow a time order
        - Other
        
        Output instruction:
        Provide your analysis in the following format:
        {{"Inference": true/false,
        "Comparison": true/false,
        "Temporal": true/false,
        "other": true/false}}

        Only respond with the JSON object, no other text needed.
        
        Question: {question}
            Options:
            A) {option_a}
            B) {option_b}
            C) {option_c}
            D) {option_d}
            Correct Answer: {correct_answer}
            Documents: {chunk_text}
        
        <end_of_turn>
        <start_of_turn>model
        """
        
        response = self.llm.invoke(prompt)
        
        try:
            # Extract the JSON string and ensure it ends with }
            # json_str = response.strip() + "}"
            # extract_index = json_str.find("}\n")
            # json_str = json_str[:extract_index + 1]
            return json.loads(response)
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


class ExamQuestionClassifier:
    def __init__(self, chunk_analyser: ChunkAnalyser):
        """Initialize the classifier with a ChunkAnalyser instance.
        
        Args:
            chunk_analyser: An instance of ChunkAnalyser class for analyzing question types
        """
        self.chunk_analyser = chunk_analyser

    def process_exam(self, exam_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process an entire exam, classifying each question and updating metadata.
        
        Args:
            exam_data: List of dictionaries containing exam questions and related data
            
        Returns:
            Updated exam data with question type classifications in metadata
        """
        updated_exam = []
        
        for question in tqdm(exam_data):
            # Create a deep copy of the question to avoid modifying the original
            question_copy = json.loads(json.dumps(question))
            
            # Get classification for the question
            classification = self.chunk_analyser.analyse_question_type(question_copy)
            
            # Update metadata with question type classification
            if "metadata" not in question_copy:
                question_copy["metadata"] = {}
            question_copy["metadata"]["question_type"] = classification
            
            updated_exam.append(question_copy)
        
        return updated_exam

    def save_exam(self, exam_data: List[Dict[str, Any]], filepath: str):
        """Save the processed exam data to a JSON file.
        
        Args:
            exam_data: List of dictionaries containing exam questions and metadata
            filepath: Path where the JSON file should be saved
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(exam_data, f, indent=2, ensure_ascii=False)


def main(input_exam_path: str, output_path: str):
    # Example usage
    chunk_analyser = ChunkAnalyser()
    classifier = ExamQuestionClassifier(chunk_analyser)
    
    # Load exam data (you would need to implement this based on your data format)
    with open(input_exam_path, 'r', encoding='utf-8') as f:
        exam_data = json.load(f)
    
    # Process the exam
    updated_exam = classifier.process_exam(exam_data)
    
    # Save the updated exam
    classifier.save_exam(updated_exam, output_path)

if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    exams = [
        "exam_new_llama_3_2_3b_processed_v5.json",
        "exam_new_ministral_8b_processed_v5.json",
        "exam_new_gemma2_9b_processed_v5.json",
        ]
    
    for task_domain in task_domains:
        for exam in exams:
            print(f"Processing: {task_domain} - {exam}")
            input_exam_path = f"MultiHopData/{task_domain}/exams/{exam}"
            exam_output_name = exam.replace("_v5.json", "_question_type.json")
            output_path = f"MultiHopData/{task_domain}/exams/{exam_output_name}"
            main(input_exam_path, output_path)
