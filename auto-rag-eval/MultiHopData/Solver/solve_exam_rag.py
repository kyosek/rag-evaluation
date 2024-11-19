from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from MultiHopData.retriever import BaseRetriever, Chunk, ChunkRetriever, RerankingRetriever
from LLMServer.llama_gcp.llama_gcp_instant import LlamaGcpModel
from LLMServer.gcp.claude_instant import ClaudeGcp
from LLMServer.gcp.gemini_instant import GeminiGcp

nltk.download("punkt_tab")


@dataclass
class ExamQuestion:
    question: str
    choices: List[str]
    correct_answer: str
    documentation: List[str]


class FAISSRetriever(BaseRetriever):
    """Dense retrieval using FAISS."""

    def __init__(self, chunk_retriever: "ChunkRetriever"):
        self.chunk_retriever = chunk_retriever

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        # Create a temporary chunk for the query
        query_chunk = Chunk(chunk_id="query", doc_id="query", content=query, original_index=-1)

        # Use the existing chunk retriever to find similar chunks
        similar_chunks = self.chunk_retriever.find_similar_chunks(
            query_chunk, k=k, similarity_threshold=0.5, exclude_same_doc=False
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
    def __init__(self, retriever: BaseRetriever, n_documents: int = 5):
        self.retriever = retriever
        self.n_documents = n_documents

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
        """Solve a single exam question using RAG with LLM."""
        retrieved_docs = self.retriever.retrieve(question.question, k=self.n_documents)

        context = "\n".join([f"{i+1}) {doc}" for i, (doc, _) in enumerate(retrieved_docs)])

        formatted_choices = "\n".join(f"{choice}" for choice in question.choices)

        # Construct a more structured prompt with system and user roles
        prompt = f"""[INST] <<SYS>>
        You are an AI assistant taking a multiple choice exam. Your task is to:
        1. Read the given question and supporting document carefully
        2. Analyze the choices
        3. Select the most appropriate answer
        4. Respond with ONLY the letter (A, B, C, or D) of the correct answer
        <</SYS>>

        Question: {question.question}

        Choices:
        {formatted_choices}
        
        Supporting documents:
        {context}

        Instructions:
        - You must respond with exactly one letter: A, B, C, or D
        - Do not include any explanation, period, or additional text
        - Just the letter of the correct answer

        Examples of valid responses:
        A
        B
        C
        D

        Your answer (one letter only): [/INST]
        """

        # Get model response
        try:
            response = model.invoke(prompt)

            # Extract just the letter from the response
            # Look for first occurrence of A, B, C, or D
            valid_answers = {"A", "B", "C", "D"}
            for char in response:
                if char in valid_answers:
                    return char

            return response.strip()[-1]
        except:
            return "A"

    def evaluate_performance(
        self, questions: List[ExamQuestion], model, task_domain, retriever_type, model_name
    ) -> Dict[str, float]:
        """Evaluate the solver's performance on a set of questions."""
        correct = 0
        total = len(questions)
        results = []

        print("Solving the exam")
        for question in tqdm(questions):
            predicted_answer = self.solve_question(question, model)

            question_result = {
                "question": question.question,
                "model_answer": predicted_answer,
                "correct_answer": question.correct_answer,
                "is_correct": predicted_answer == question.correct_answer,
            }

            # Add the question result to the list
            results.append(question_result)

            if predicted_answer == question.correct_answer:
                correct += 1

        metrics = {"accuracy": correct / total, "correct": correct, "total": total}

        with open(
            f"MultiHopData/{task_domain}/{model_name}_{retriever_type}_exam_results.json", "w"
        ) as json_file:
            json.dump(results, json_file, indent=2)

        return metrics


def main(
    task_domain: str, retriever_type: str, model_type: str, model_name: str, reranking: bool = False
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

    solver = ExamSolver(retriever)

    # Load and solve exam
    if model_type == "gemini":
        model = GeminiGcp(model_name=model_name)
    elif model_type == "claude":
        model = ClaudeGcp(model_name=model_name)
    else:
        print("Using Llama-cpp")
        # model = LlamaModel(model_path=model_path)

    questions = solver.load_exam(f"MultiHopData/{task_domain}/exam_cleaned_1000_42.json")
    metrics = solver.evaluate_performance(questions, model, task_domain, retriever_type, model_name)

    print(f"Exam Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")


if __name__ == "__main__":
    # Model family
    # model_type = "gemini"
    model_type = "claude"

    # Task domain
    # task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    task_domains = ["gov_report"]

    # Retriever type
    retriever_types = ["Dense", "Sparse", "Hybrid"]
    # retriever_types = ["Dense", "Hybrid"]

    # Model name
    # model_names = ["gemini-1.5-pro-002", "gemini-1.5-flash-002"]
    # model_names = ["claude-3-5-sonnet@20240620", "claude-3-5-haiku@20241022"]
    model_names = ["claude-3-5-haiku@20241022"]
    
    # Reranker flag
    # rerank_flags = [True, False]
    rerank_flags = [False]

    for rerank_flag in rerank_flags:
        for model_name in model_names:
            for task_domain in task_domains:
                for retriever_type in retriever_types:
                    print(f"Using {model_name}")
                    print(f"Processing {task_domain}")
                    print(f"Retriever: {retriever_type}")
                    print(f"Rerank: {rerank_flag}")
                    main(task_domain, retriever_type, model_type, model_name, reranking=rerank_flag)
