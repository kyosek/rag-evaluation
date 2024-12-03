import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import faiss
import numpy as np
from tqdm import tqdm

from MultiHopData.retriever import BaseRetriever, BM25Retriever, Chunk, ChunkRetriever, FAISSRetriever, HybridRetriever, RerankingRetriever
from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.gcp.gemini_instant import GeminiGcp
from LLMServer.llama.llama_instant import ModelFactory, ModelType


@dataclass
class ExamQuestion:
    question: str
    choices: List[str]
    correct_answer: str
    documentation: List[str]
    retrieved_chunks: Optional[Dict[str, List[Dict[str, Union[str, float]]]]] = None


class ExamSolver:
    def __init__(self, retriever: Optional[BaseRetriever] = None, n_documents: int = 15):
        self.retriever = retriever
        self.n_documents = n_documents

    def load_exam(self, exam_file: str) -> List[ExamQuestion]:
        """Load exam questions from JSON file with pre-retrieved chunks."""
        with open(exam_file, "r") as f:
            data = json.load(f)

        questions = []
        for item in data:
            question = ExamQuestion(
                question=item["question"],
                choices=item["choices"],
                correct_answer=item["correct_answer"],
                documentation=item.get("documentation", []),
                retrieved_chunks=item.get("retrieved_chunks", None)
            )
            questions.append(question)
        return questions

    def solve_question(self, question: ExamQuestion, retriever_method: str, model) -> str:
        """Solve a single exam question using either live retrieval or pre-retrieved chunks."""
        if question.retrieved_chunks and retriever_method in question.retrieved_chunks:
            # Use pre-retrieved chunks
            retrieved_docs = question.retrieved_chunks[retriever_method]
            # Sort by score in descending order and take top n_documents
            sorted_docs = sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)[:self.n_documents]
            context = "\n".join([f"{i+1}) {doc['content']}" for i, doc in enumerate(sorted_docs)])
        elif self.retriever:
            # Fallback to live retrieval if no pre-retrieved chunks available
            retrieved_docs = self.retriever.retrieve(question.question, k=self.n_documents)
            context = "\n".join([f"{i+1}) {doc}" for i, (doc, _) in enumerate(retrieved_docs)])
        else:
            context = "No supporting documents available."

        formatted_choices = "\n".join(f"{choice}" for choice in question.choices)

        prompt = f"""<s>[INST] <<SYS>>
        You are an AI assistant taking a multiple choice exam.
        Your task is to:
        1. Read the given question, choices and supporting document carefully
        2. Select the most appropriate answer
        3. Respond with ONLY one letter (A, B, C, or D) of the correct answer
        
        Instructions:
        - You must respond with exactly one letter: A, B, C, or D
        - Do not include any explanation, period, or additional text
        - Just the letter of the correct answer

        Examples of valid responses:
        A
        B
        C
        D

        <</SYS>>[/INST]

        Question: {question.question}

        Choices:
        {formatted_choices}
        
        Supporting documents:
        {context}
        
        Your answer (one letter only):
        </s>
        """

        try:
            response = model.invoke(prompt)
            valid_answers = {"A", "B", "C", "D"}
            for char in response:
                if char in valid_answers:
                    return char
            return response.strip()[-1]
        except:
            return "NA"

    def evaluate_performance(
        self, questions: List[ExamQuestion], model, task_domain, retriever_type, model_name, exam_file
    ) -> Dict[str, float]:
        """Evaluate the solver's performance on a set of questions."""
        correct = 0
        total = len(questions)
        results = []

        print(f"Solving the exam using {retriever_type} retriever")
        for question in tqdm(questions):
            # Map retriever type to the corresponding key in retrieved_chunks
            if retriever_type != "Rerank":
                retriever_method = retriever_type.lower()  # 'Dense' -> 'dense', etc.
            predicted_answer = self.solve_question(question, retriever_method, model)

            question_result = {
                "question": question.question,
                "model_answer": predicted_answer,
                "correct_answer": question.correct_answer,
                "is_correct": predicted_answer == question.correct_answer,
                "number_of_hops": len(question.documentation)
            }

            results.append(question_result)

            if predicted_answer == question.correct_answer:
                correct += 1

        metrics = {"accuracy": correct / total, "correct": correct, "total": total}

        # Save results
        results_dir = f"MultiHopData/{task_domain}/exam_results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(
            results_dir,
            f"{model_name}_{retriever_type}_{os.path.basename(exam_file)}_results.json"
        )
        
        with open(results_path, "w") as json_file:
            json.dump(results, json_file, indent=2)

        return metrics


def main(
    task_domain: str, retriever_type: str, model_type: str, model_name: str, exam_file: str, n_documents: int, reranking: bool = False
):
    chunk_retriever = ChunkRetriever(task_domain, random_seed=42)

    chunk_retriever = chunk_retriever.load_database(
        f"MultiHopData/{task_domain}/chunk_database", task_domain
    )

    # Initialize different retrievers
    faiss_retriever = FAISSRetriever(chunk_retriever)
    bm25_retriever = BM25Retriever([chunk.content for chunk in chunk_retriever.chunks])

    # Create a hybrid retriever
    hybrid_retriever = HybridRetriever([(faiss_retriever, 0.5), (bm25_retriever, 0.5)])

    # Initialize solver with chosen retriever
    if retriever_type == "Dense":
        retriever = faiss_retriever
    elif retriever_type == "Sparse":
        retriever = bm25_retriever
    elif retriever_type == "Hybrid":
        retriever = hybrid_retriever

    if reranking:
        retriever = RerankingRetriever(retriever)

    solver = ExamSolver(retriever, n_documents)

    # Load and solve exam
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

    questions = solver.load_exam(f"MultiHopData/{task_domain}/exams/{exam_file}")
    metrics = solver.evaluate_performance(questions, model, task_domain, retriever_type, model_name, exam_file)

    print(f"Exam Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")


if __name__ == "__main__":
    # Model family
    # model_type = "gemini"
    # model_type = "claude"
    model_type = "cpp"

    # Task domain
    # task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    task_domains = ["gov_report"]

    # Retriever type
    retriever_types = ["Dense", "Sparse", "Hybrid", "Rerank"]
    # retriever_types = ["Dense", "Hybrid"]

    # Model name
    # model_names = ["gemini-1.5-pro-002", "gemini-1.5-flash-002"]
    # model_names = ["claude-3-5-sonnet@20240620", "claude-3-5-haiku@20241022"]
    # model_names = ["claude-3-5-haiku@20241022"]
    model_names = [
        'llama_3_2_3b',
        "ministral-8b",
        "gemma2-27b",
        ]
    
    # Exam file
    exam_files = [
        "llama_3_1_8b_single_hop_exam_cleaned_shuffled_1000_42.json",
        "llama_3_2_3b_single_hop_exam_cleaned_shuffled_1000_42.json"
        ]

    for exam_file in exam_files:
        for model_name in model_names:
            for task_domain in task_domains:
                for retriever_type in retriever_types:
                    print(f"Using {model_name}")
                    print(f"Solving {exam_file} of {task_domain}")
                    print(f"Retriever: {retriever_type}")
                    main(task_domain, retriever_type, model_type, model_name, exam_file, n_documents=15)
