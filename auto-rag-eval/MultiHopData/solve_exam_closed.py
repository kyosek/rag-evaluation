from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import json
import torch
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

from MultiHopData.generate_exam import ChunkRetriever, Chunk
from LLMServer.llama.llama_instant import LlamaModel
from LLMServer.llama_gcp.llama_gcp_instant import LlamaGcpModel
from LLMServer.gcp.claude_instant import ClaudeGcp

model_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "device_map": "auto"
}


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
    def __init__(self, chunk_retriever: 'ChunkRetriever'):
        self.chunk_retriever = chunk_retriever
        
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        # Create a temporary chunk for the query
        query_chunk = Chunk(
            chunk_id="query",
            doc_id="query",
            content=query,
            original_index=-1
        )
        
        # Use the existing chunk retriever to find similar chunks
        similar_chunks = self.chunk_retriever.find_similar_chunks(
            query_chunk,
            k=k,
            exclude_same_doc=False
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
        with open(exam_file, 'r') as f:
            data = json.load(f)
            
        questions = []
        for item in data:
            question = ExamQuestion(
                question=item['question'],
                choices=item['choices'],
                correct_answer=item['correct_answer'],
                documentation=item.get('documentation', [])
            )
            questions.append(question)
        return questions
    
    def solve_question(self, question: ExamQuestion, model) -> str:
        """Solve a single exam question with LLM."""
        
        formatted_choices = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(question.choices))
    
        # Construct a more structured prompt with system and user roles
        prompt = f"""[INST] <<SYS>>
        You are an AI assistant taking a multiple choice exam. Your task is to:
        1. Read the question and provided document carefully
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

        Your answer (one letter only): [/INST]"""

        # Get model response
        response = model.invoke(prompt)
        
        # Extract just the letter from the response
        # Look for first occurrence of A, B, C, or D
        valid_answers = {'A', 'B', 'C', 'D'}
        for char in response:
            if char in valid_answers:
                return char
            
        # If no valid letter found, return the last character as fallback
        try:
            return response.strip()[-1]
        except:
            return "A"
            
            return answer
    
    def evaluate_performance(self, questions: List[ExamQuestion], model) -> Dict[str, float]:
        """Evaluate the solver's performance on a set of questions."""
        correct = 0
        total = len(questions)
        
        for question in tqdm(questions):
            predicted_answer = self.solve_question(question, model)
            if predicted_answer == question.correct_answer:
                correct += 1
        
        metrics = {
            'accuracy': correct / total,
            'correct': correct,
            'total': total
        }
        return metrics


def main(task_domain: str, retriever_type: str, model_name: str):
    if model_name == "GCP":
        print("Using transformer")
        model = LlamaGcpModel(
            model_size="70B",
            use_gpu=True,
            model_config=model_config,
            )
    elif model_name == "claude":
        model = ClaudeGcp()
    else:
        print("Using Llama-cpp")
        # model = LlamaModel(model_path=model_path)
    
    solver = ExamSolver()
    questions = solver.load_exam(f"MultiHopData/{task_domain}/exam_cleaned_1000_42.json")
    metrics = solver.evaluate_performance(questions, model)
    
    print(f"Exam Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")


if __name__ == "__main__":
    task_domain = "SecFilings"
    retriever_type = "Dense"
    model_name = "claude"
    
    main(task_domain, retriever_type, model_name)
