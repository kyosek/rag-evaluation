import os
import json
import pickle
import random
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from sentence_transformers import CrossEncoder

from LLMServer.gcp.gemini_instant import GeminiGcp
from LLMServer.gcp.claude_instant import ClaudeGcp
from MultiHopData.retriever import Chunk, BaseRetriever, RerankingRetriever
from MultiHopData.solve_exam_rag import ExamQuestion, ExamSolver

nltk.download("punkt_tab")


@dataclass
class ContextualChunk:
    chunk_id: str
    doc_id: str
    content: str
    original_index: int
    context: str = ""


class ContextualChunkRetriever:
    def __init__(
        self,
        task_domain: str,
        llm_model_name: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        random_seed: Optional[int] = None,
    ):
        self.model = SentenceTransformer(embedding_model_name)
        self.task_domain = task_domain
        self.index = None
        self.chunks: List[ContextualChunk] = []
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

        self.claude = ClaudeGcp(model_name=llm_model_name)
        self.gemini = GeminiGcp(model_name=llm_model_name)

    def generate_context(self, doc_content: str, chunk_content: str, model) -> str:
        prompt = f"""
        <document>
        {doc_content}
        </document>

        <chunk>
        {chunk_content}
        </chunk>

        Create a concise a few sentences summary that:
        1. Describes how this chunk fits into the broader document
        2. Captures key relationships to other document sections
        3. Identifies the main technical concepts/terms
        4. Notes any dependencies or prerequisites mentioned
        5. Highlights unique identifying details

        Focus on information that would help distinguish this chunk from similar content.
        Be specific but concise.
        Avoid generic descriptions. Include technical terms that someone might search for.

        Output only the contextual summary, with no additional text or explanations.
        """
        response = model.invoke(prompt)
        return response.strip()

    def load_documents(self, json_file: str) -> None:
        with open(json_file, "r") as f:
            data = json.load(f)

        for doc in tqdm(data, desc="Processing documents"):
            doc_id = doc["doc_id"]
            doc_content = "\n".join([chunk["content"] for chunk in doc["chunks"]])

            for chunk in doc["chunks"]:
                try:
                    context = self.generate_context(doc_content, chunk["content"], self.llama)
                except:
                    print("Generating with Claude")
                    context = self.generate_context(doc_content, chunk["content"], self.claude)
                chunk_obj = ContextualChunk(
                    chunk_id=chunk["chunk_id"],
                    doc_id=doc_id,
                    content=chunk["content"],
                    original_index=chunk["original_index"],
                    context=context,
                )
                self.chunks.append(chunk_obj)

        self._build_index()

    def _build_index(self) -> None:
        embeddings = self.model.encode(
            [f"{chunk.content}\n\nContext: {chunk.context}" for chunk in self.chunks]
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def find_similar_chunks(
        self,
        query_chunk: ContextualChunk,
        k: int = 4,
        similarity_threshold: float = 0.5,
        exclude_same_doc: bool = True,
    ) -> List[Tuple[ContextualChunk, float]]:
        query_embedding = self.model.encode(
            [f"{query_chunk.content}\n\nContext: {query_chunk.context}"], normalize_embeddings=True
        )

        scores, indices = self.index.search(query_embedding, k * 2)

        similar_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score < similarity_threshold:
                continue

            chunk = self.chunks[idx]
            if exclude_same_doc and chunk.doc_id == query_chunk.doc_id:
                continue

            if chunk.chunk_id != query_chunk.chunk_id:
                similar_chunks.append((chunk, float(score)))

            if len(similar_chunks) >= k:
                break

        return similar_chunks

    def save_database(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))

        metadata = {"chunks": self.chunks, "random_seed": self.random_seed}
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load_database(
        cls,
        directory: str,
        llm_model_name: str,
        task_domain: str,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> "ContextualChunkRetriever":
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        instance = cls(
            task_domain,
            llm_model_name=llm_model_name,
            embedding_model_name=model_name,
            random_seed=metadata["random_seed"],
        )
        instance.chunks = metadata["chunks"]

        instance.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))

        if instance.index is None:
            instance._build_index()

        return instance


class ContextualFAISSRetriever(BaseRetriever):
    def __init__(self, chunk_retriever: ContextualChunkRetriever):
        self.chunk_retriever = chunk_retriever

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        query_chunk = ContextualChunk(
            chunk_id="query",
            doc_id="query",
            content=query,
            original_index=-1,
            context="",  # We don't generate context for the query
        )

        similar_chunks = self.chunk_retriever.find_similar_chunks(
            query_chunk, k=k, exclude_same_doc=False
        )

        return [
            (f"{chunk.content}\n\nContext: {chunk.context}", score)
            for chunk, score in similar_chunks
        ]


class ContextualBM25Retriever(BaseRetriever):
    def __init__(self, chunks: List[ContextualChunk]):
        self.chunks = chunks
        self.corpus = [f"{chunk.content}\n\nContext: {chunk.context}" for chunk in chunks]
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        tokenized_query = word_tokenize(query.lower())
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[-k:][::-1]
        return [(self.corpus[i], doc_scores[i]) for i in top_indices]


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever: ContextualFAISSRetriever,
        sparse_retriever: ContextualBM25Retriever,
        dense_weight: float = 0.5,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = 1 - dense_weight

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        dense_results = self.dense_retriever.retrieve(query, k=k)
        sparse_results = self.sparse_retriever.retrieve(query, k=k)

        combined_results = {}
        for doc, score in dense_results:
            combined_results[doc] = score * self.dense_weight
        for doc, score in sparse_results:
            if doc in combined_results:
                combined_results[doc] += score * self.sparse_weight
            else:
                combined_results[doc] = score * self.sparse_weight

        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]


class ContextualExamSolver(ExamSolver):
    def __init__(self, retriever: BaseRetriever, n_documents: int = 5):
        super().__init__(retriever, n_documents)

    def solve_question(self, question: ExamQuestion, model) -> str:
        retrieved_docs = self.retriever.retrieve(question.question, k=self.n_documents)

        context = "\n".join([f"{i+1}) {doc}" for i, (doc, _) in enumerate(retrieved_docs)])

        formatted_choices = "\n".join(f"{choice}" for choice in question.choices)

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

        try:
            response = model.invoke(prompt)

            valid_answers = {"A", "B", "C", "D"}
            for char in response:
                if char in valid_answers:
                    return char

            return response.strip()[-1]
        except:
            return "NA"


def main(
    task_domain: str, retriever_type: str, model_type: str, model_name: str, reranking: bool = False
):
    chunk_retriever = ContextualChunkRetriever(
        task_domain, llm_model_name=model_name, random_seed=42
    )

    # Load or create the contextual database
    db_path = f"MultiHopData/{task_domain}/contextual_chunk_flash_database"
    if os.path.exists(db_path):
        chunk_retriever = chunk_retriever.load_database(db_path, model_name, task_domain)
    else:
        chunk_retriever.load_documents(f"MultiHopData/{task_domain}/docs_chunk.json")
        chunk_retriever.save_database(db_path)

    # Initialize solver with contextual retriever
    contextual_faiss_retriever = ContextualFAISSRetriever(chunk_retriever)
    bm25_retriever = ContextualBM25Retriever(chunk_retriever.chunks)

    if retriever_type == "Dense":
        retriever = contextual_faiss_retriever
    elif retriever_type == "Sparse":
        retriever = bm25_retriever
    elif retriever_type == "Hybrid":
        retriever = HybridRetriever(contextual_faiss_retriever, bm25_retriever)
    else:
        raise ValueError("Invalid retriever type")

    if reranking:
        retriever = RerankingRetriever(retriever)

    solver = ContextualExamSolver(retriever)

    # Load and solve exam
    if model_type == "claude":
        model = ClaudeGcp(model_name=model_name)
    elif model_type == "gemini":
        model = GeminiGcp(model_name=model_name)
    else:
        print("Not a valid model name")

    questions = solver.load_exam(f"MultiHopData/{task_domain}/exam_cleaned_1000_42.json")
    metrics = solver.evaluate_performance(questions, model)

    print(f"Exam Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")


if __name__ == "__main__":
    # Model family
    # model_type = "gemini"
    model_type = "claude"
    
    # Task domain
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]

    # Retriever type
    retriever_types = ["Dense", "Sparse", "Hybrid"]
    # retriever_types = ["Sparse"]
    
    # Model name
    # model_names = ["gemini-1.5-pro-002", "gemini-1.5-flash-002"]
    # model_names = ["claude-3-5-sonnet@20240620", "claude-3-5-haiku@20241022"]
    model_names = ["claude-3-5-haiku@20241022"]
    
    # Reranker flag
    # rerank_flags = [False, True]
    rerank_flags = [True]

    for rerank_flag in rerank_flags:
        for model_name in model_names:
            for task_domain in task_domains:
                for retriever_type in retriever_types:
                    print(f"Using {model_name}")
                    print(f"Processing {task_domain}")
                    print(f"Retriever: {retriever_type}")
                    print(f"Rerank: {rerank_flag}")
                    main(task_domain, retriever_type, model_type, model_name, reranking=rerank_flag)
                # main(task_domain, retriever_type, model_type, model_name)
