from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import json
import faiss
import numpy as np
from tqdm import tqdm

from MultiHopData.retriever import Chunk, ChunkRetriever
from LLMServer.llama_gcp.llama_gcp_instant import LlamaGcpModel
from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.gcp.gemini_instant import GeminiGcp
from LLMServer.llama.llama_instant import ModelFactory, ModelType


@dataclass
class ExamQuestion:
    question: str
    choices: List[str]
    correct_answer: str
    documentation: List[str]


class ExamSolver:
    def load_exam(self, exam_file: str) -> List[ExamQuestion]:
        """Load exam questions from JSON file."""
        with open(exam_file, "r") as f:
            data = json.load(f)

        questions = []
        for item in data:
            question = ExamQuestion(
                question=item["question"],
                choices=item["choices"],
                correct_answer=item["correct_answer"],
                documentation=item.get("documentation", []),
            )
            questions.append(question)
        return questions

    def solve_question(self, question: ExamQuestion, model) -> str:
        """Solve a single exam question with LLM."""

        formatted_choices = "\n".join(
            f"{chr(65+i)}. {choice}" for i, choice in enumerate(question.choices)
        )

        # Construct a more structured prompt with system and user roles
        prompt = f"""<s>[INST] <<SYS>>
        You are an AI assistant taking a multiple choice exam. Your task is to:
        1. Read the question, provided choices and documents carefully
        2. Analyze the choices
        3. The correct answer must be found in the given documents
        4. Select the most appropriate answer
        5. Respond with ONLY the letter (A, B, C, or D) of the correct answer
        <</SYS>>

        Question: {question.question}

        Choices:
        {formatted_choices}
        
        Documents:
        {question.documentation}

        Instructions:
        - You must respond with exactly one letter: A, B, C, or D
        - Do not include any explanation, period, or additional text
        - Just the letter of the correct answer

        Examples of valid responses:
        A
        B
        C
        D

        Your answer (one letter only): [/INST]</s>"""

        # Get model response
        try:
            response = model.invoke(prompt)

            # Extract just the letter from the response
            # Look for first occurrence of A, B, C, or D
            valid_answers = {"A", "B", "C", "D"}
            for char in response:
                if char in valid_answers:
                    return char

            # If no valid letter found, return the last character as fallback
            return response.strip()[-1]
        except:
            return "NA"

    def evaluate_performance(
        self, questions: List[ExamQuestion], model, task_domain, model_name, exam_file
    ) -> Dict[str, float]:
        """Evaluate the solver's performance on a set of questions."""
        correct = 0
        total = len(questions)
        results = []

        for question in tqdm(questions):
            predicted_answer = self.solve_question(question, model)

            question_result = {
                "question": question.question,
                "model_answer": predicted_answer,
                "correct_answer": question.correct_answer,
                "is_correct": predicted_answer == question.correct_answer,
                "number_of_hops": len(question.documentation)
            }

            # Add the question result to the list
            results.append(question_result)

            if predicted_answer == question.correct_answer:
                correct += 1

        metrics = {"accuracy": correct / total, "correct": correct, "total": total}

        with open(f"MultiHopData/{task_domain}/exam_results/{model_name}_open_{exam_file}.json", "w") as json_file:
            json.dump(results, json_file, indent=2)

        return metrics


def main(task_domain: str, model_type: str, model_name: str, exam_file: str):
    if model_type == "gemini":
        model = GeminiGcp(model_name=model_name)
    elif model_type == "claude":
        model = ClaudeGcp(model_name=model_name)
    elif model_type == "cpp":
        model_mapping = {
            'llama_3_1_8b': ModelType.LLAMA_3_1_8B,
            'llama_3_2_3b': ModelType.LLAMA_3_2_3B,
            'ministral-8b': ModelType.MINISTRAL_8B,
            'mistral-small': ModelType.MISTRAL_SMALL,
            'mixtral-8-7b': ModelType.MIXTRAL_8_7B,
            "gemma2-9b": ModelType.GEMMA2_9B,
            "gemma2-27b": ModelType.GEMMA2_27B
        }
        
        print(f"Using {model_mapping[model_name]}")
        model = ModelFactory.create_model(model_mapping[model_name])
    else:
        print("Using Llama-cpp")
        # model = LlamaModel(model_path=model_path)

    print("Solving the exam")
    solver = ExamSolver()
    questions = solver.load_exam(f"MultiHopData/{task_domain}/exams/{exam_file}")
    metrics = solver.evaluate_performance(questions, model, task_domain, model_name, exam_file)

    print(f"Exam Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")


if __name__ == "__main__":
    # task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    task_domains = ["wiki"]
    # model_type = "claude"
    # model_type = "gemini"
    model_type = "cpp"
    # model_name = "claude-3-5-haiku@20241022"
    # model_name = "claude-3-5-sonnet@20240620"
    # model_name = "gemini-1.5-pro-002"
    # model_name = "gemini-1.5-flash-002"

    # model_names = ["gemini-1.5-pro-002", "gemini-1.5-flash-002"]
    # model_names = ["claude-3-5-sonnet@20240620", "claude-3-5-haiku@20241022"]
    # model_names = ["claude-3-5-sonnet@20240620"]
    model_names = [
        # 'llama_3_1_8b',
        # "ministral-8b",
        "gemma2-9b",
        ]
    
    exam_files = [
        # "llama_3_2_3b_single_hop_exam_processed.json",
        # "gemma2_9b_single_hop_exam_processed.json",
        # "ministral_8b_single_hop_exam_processed.json",
        "exam_new_llama_3_2_3b_processed_v3.json",
        # "exam_new_gemma2_9b_processed_v2.json",
        # "exam_new_ministral_8b_processed_v3.json",
        ]

    for exam_file in exam_files:
        for model_name in model_names:
            for task_domain in task_domains:
                print(f"Using {model_name}")
                print(f"Solving {exam_file} on {task_domain}")
                main(task_domain, model_type, model_name, exam_file)
