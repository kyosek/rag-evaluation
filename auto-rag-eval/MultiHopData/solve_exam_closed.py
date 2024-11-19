import re
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import json
import torch
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

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


class BaseRetriever(ABC):
    """Abstract base class for different retrieval methods."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query."""
        pass


class FAISSRetriever(BaseRetriever):
    """Dense retrieval using FAISS."""

    def __init__(self, chunk_retriever: "ChunkRetriever"):
        self.chunk_retriever = chunk_retriever

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        # Create a temporary chunk for the query
        query_chunk = Chunk(chunk_id="query", doc_id="query", content=query, original_index=-1)

        # Use the existing chunk retriever to find similar chunks
        similar_chunks = self.chunk_retriever.find_similar_chunks(
            query_chunk, k=k, exclude_same_doc=False
        )

        return [(chunk.content, score) for chunk, score in similar_chunks]


class BM25Retriever(BaseRetriever):
    """Sparse retrieval using BM25."""

    def __init__(self, documents: List[str]):
        # Tokenize documents
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[-k:][::-1]

        return [(self.documents[i], scores[i]) for i in top_k_indices]


class HybridRetriever(BaseRetriever):
    """Combines multiple retrievers with optional weights."""

    def __init__(self, retrievers: List[Tuple[BaseRetriever, float]]):
        self.retrievers = retrievers  # List of (retriever, weight) tuples

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        all_results = []

        # Get results from each retriever
        for retriever, weight in self.retrievers:
            results = retriever.retrieve(query, k=k)
            weighted_results = [(doc, score * weight) for doc, score in results]
            all_results.extend(weighted_results)

        # Combine and deduplicate results
        unique_results = {}
        for doc, score in all_results:
            if doc in unique_results:
                unique_results[doc] = max(unique_results[doc], score)
            else:
                unique_results[doc] = score

        # Sort by score and return top k
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]


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
        1. Read the question and choices carefully
        2. Analyze the choices
        3. Select the most appropriate answer
        4. Respond with ONLY the letter (A, B, C, or D) of the correct answer
        <</SYS>>

        Question: {question.question}

        Choices:
        {formatted_choices}

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
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize the model name for use in filenames."""
        # Replace forward slashes with underscores
        # Remove or replace other potentially problematic characters
        sanitized = re.sub(r'[/\\:*?"<>|]', '_', filename)
        return sanitized

    def evaluate_performance(
        self, questions: List[ExamQuestion], model, task_domain, model_name
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

        results_dir = os.path.join("MultiHopData", task_domain)
        os.makedirs(results_dir, exist_ok=True)

        # Sanitize the model name for the filename
        safe_model_name = self.sanitize_filename(model_name)
        results_file = os.path.join(results_dir, f"{safe_model_name}_closed_exam_results.json")
        with open(results_file, "w") as json_file:
            json.dump(results, json_file, indent=2)

        return metrics


def main(task_domain: str, model_type: str, model_name: str):
    if model_type == "gemini":
        model = GeminiGcp(model_name=model_name)

    elif model_type == "claude":
        model = ClaudeGcp(model_name=model_name)
    elif model_type == "cpp":
        model = ModelFactory.create_model(ModelType.LLAMA_3_2_3B)
        # model = ModelFactory.create_model(ModelType.MINISTRAL_8B)
        # model = ModelFactory.create_model(ModelType.LLAMA_3_1_8B)
    else:
        print("Invalid model name")

    print("Solving the exam")
    solver = ExamSolver()
    questions = solver.load_exam(f"MultiHopData/{task_domain}/exams/exam_new_cleaned_1000_42_llama3_3b.json")
    metrics = solver.evaluate_performance(questions, model, task_domain, model_name)

    print(f"Exam Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")


if __name__ == "__main__":
    # task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    task_domains = ["gov_report"]
    # model_type = "claude"
    # model_type = "gemini"
    model_type = "cpp"
    # model_name = "claude-3-5-haiku@20241022"
    # model_name = "claude-3-5-sonnet@20240620"
    # model_name = "gemini-1.5-pro-002"
    # model_name = "gemini-1.5-flash-002"
    model_name = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

    # model_names = ["MINISTRAL_8B"]
    model_names = ["LLAMA_3_2_3B"]
    # model_names = ["gemini-1.5-pro-002", "gemini-1.5-flash-002"]

    for model_name in model_names:
        for task_domain in task_domains:
            print(f"Using {model_name}")
            print(f"Solving {task_domain}")
            main(task_domain, model_type, model_name)
