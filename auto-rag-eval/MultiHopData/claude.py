from LLMServer.gcp.claude_instant import ClaudeGcp

model = ClaudeGcp()

prompt = """I am developing python scripts to solve multiple choice exam by RAG system.

I would like to implement an evaluation metrics to evaluate how good/bad is retrieval but without running the generation part of the RAG.

I've seen this function to evaluate that.

def evaluate_db(db, original_jsonl_path: str, k):
    # Load the original JSONL data for queries and ground truth
    original_data = load_jsonl(original_jsonl_path)
    
    # Evaluate retrieval
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    print(f"Total Score: {results['average_score']}")
    print(f"Total queries: {results['total_queries']}")
    
My current script looks like:

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

nltk.download('punkt_tab')


@dataclass
class ExamQuestion:
    question: str
    choices: List[str]
    correct_answer: str
    documentation: List[str]


class FAISSRetriever(BaseRetriever):
    ""Dense retrieval using FAISS.""
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
            similarity_threshold=0.5,
            exclude_same_doc=False
        )
        
        return [(chunk.content, score) for chunk, score in similar_chunks]


class BM25Retriever(BaseRetriever):
    ""Sparse retrieval using BM25.""
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
    "Combines multiple retrievers with optional weights.""
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
        ""Load exam questions from JSON file.""
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
        ""Solve a single exam question using RAG with LLM.""
        retrieved_docs = self.retriever.retrieve(question.question, k=self.n_documents)
        
        context = "\n".join([f"{i+1}) {doc}" for i, (doc, _) in enumerate(retrieved_docs)])
        
        formatted_choices = "\n".join(f"{choice}" for choice in question.choices)
    
        # Construct a more structured prompt with system and user roles
        prompt = f""[INST] <<SYS>>
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
        ""

        # Get model response
        try:
            response = model.invoke(prompt)
            
            # Extract just the letter from the response
            # Look for first occurrence of A, B, C, or D
            valid_answers = {'A', 'B', 'C', 'D'}
            for char in response:
                if char in valid_answers:
                    return char
                
            return response.strip()[-1]
        except:
            return "A"
    
    def evaluate_performance(self, questions: List[ExamQuestion], model) -> Dict[str, float]:
        ""Evaluate the solver's performance on a set of questions.""
        correct = 0
        total = len(questions)
        
        print("Solving the exam")
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


def main(task_domain: str, retriever_type: str, model_type: str, model_name: str, reranking: bool = False):
    chunk_retriever = ChunkRetriever(task_domain, random_seed=42)
    
    chunk_retriever = chunk_retriever.load_database(f"MultiHopData/{task_domain}/chunk_database", task_domain)
    
    # Initialize different retrievers
    faiss_retriever = FAISSRetriever(chunk_retriever)
    bm25_retriever = BM25Retriever([chunk.content for chunk in chunk_retriever.chunks])
    
    # Create a hybrid retriever
    hybrid_retriever = HybridRetriever([
        (faiss_retriever, 0.5),
        (bm25_retriever, 0.5)
    ])
    
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
    # retriever_types = ["Dense", "Hybrid"]
    
    # Model name
    # model_names = ["gemini-1.5-pro-002", "gemini-1.5-flash-002"]
    model_names = ["claude-3-5-sonnet@20240620", "claude-3-5-haiku@20241022"]
    
    # Reranker flag
    rerank_flags = [True, False]
    
    for rerank_flag in rerank_flags:
        for model_name in model_names:
            for task_domain in task_domains:
                for retriever_type in retriever_types:
                    print(f"Using {model_name}")            
                    print(f"Processing {task_domain}")
                    print(f"Retriever: {retriever_type}")
                    print(f"Rerank: {rerank_flag}")
                    main(task_domain, retriever_type, model_type, model_name, reranking=rerank_flag)


from abc import ABC, abstractmethod
import json
import faiss
import random
import os
import pickle

from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from sentence_transformers import CrossEncoder


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    original_index: int


class BaseRetriever(ABC):
    ""Abstract base class for different retrieval methods.""
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        ""Retrieve relevant documents for a query.""
        pass


class ChunkRetriever:
    def __init__(self, task_domain: str, model_name: str = 'all-MiniLM-L6-v2', random_seed: Optional[int] = None):
        ""
        Initialize the chunk retriever with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            random_seed: Seed for random operations (optional)
        ""
        self.model = SentenceTransformer(model_name)
        self.task_domain = task_domain
        self.index = None
        self.chunks: List[Chunk] = []
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
        
    def load_documents(self, json_file: str) -> None:
        ""
        Load documents from JSON file and store chunks.
        
        Args:
            json_file: Path to the JSON file containing documents
        ""
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Extract all chunks from the documents
        for doc in data:
            doc_id = doc['doc_id']
            for chunk in doc['chunks']:
                chunk_obj = Chunk(
                    chunk_id=chunk['chunk_id'],
                    doc_id=doc_id,
                    content=chunk['content'],
                    original_index=chunk['original_index']
                )
                self.chunks.append(chunk_obj)
        
        # Create FAISS index
        self._build_index()
        
    def _build_index(self) -> None:
        ""Build FAISS index from chunks.""
        # Generate embeddings for all chunks
        embeddings = self.model.encode([chunk.content for chunk in self.chunks])
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product is equivalent to cosine similarity for normalized vectors
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add vectors to the index
        self.index.add(embeddings)
        
    def sample_chunks(self, n: int, seed: Optional[int] = None) -> List[Chunk]:
        "
        Randomly sample n chunks from the dataset.
        
        Args:
            n: Number of chunks to sample
            seed: Random seed for sampling (overrides instance seed if provided)
            
        Returns:
            List of sampled chunks
        ""
        if seed is not None:
            # Temporarily set seed for this operation
            current_state = random.getstate()
            random.seed(seed)
            samples = random.sample(self.chunks, min(n, len(self.chunks)))
            random.setstate(current_state)
            return samples
        return random.sample(self.chunks, min(n, len(self.chunks)))
    
    def find_similar_chunks(
        self, 
        query_chunk: Chunk, 
        k: int = 4, 
        similarity_threshold: float = 0.9,
        exclude_same_doc: bool = True
    ) -> List[Tuple[Chunk, float]]:
        "
        Find similar chunks for a given query chunk.
        
        Args:
            query_chunk: The chunk to find similar chunks for
            k: Number of similar chunks to retrieve
            similarity_threshold: Minimum similarity score threshold
            exclude_same_doc: Whether to exclude chunks from the same document
            
        Returns:
            List of tuples containing similar chunks and their similarity scores
        "
        # Generate embedding for query chunk
        query_embedding = self.model.encode([query_chunk.content])
        faiss.normalize_L2(query_embedding)
        
        # Search in the index
        scores, indices = self.index.search(query_embedding, k * 2)  # Get more results initially for filtering
        
        # Filter and process results
        similar_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score < similarity_threshold:
                continue
                
            chunk = self.chunks[idx]
            if exclude_same_doc and chunk.doc_id == query_chunk.doc_id:
                continue
                
            if chunk.chunk_id != query_chunk.chunk_id:  # Exclude the query chunk itself
                similar_chunks.append((chunk, float(score)))
                
            if len(similar_chunks) >= k:
                break
                
        return similar_chunks

    def save_database(self, directory: str) -> None:
        "
        Save the database (FAISS index and chunks) to disk.
        
        Args:
            directory: Directory to save the database files
        "
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
        
        # Save chunks and other metadata
        metadata = {
            'chunks': self.chunks,
            'random_seed': self.random_seed
        }
        with open(os.path.join(directory, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
            
    @classmethod
    def load_database(cls, directory: str, task_domain: str, model_name: str = 'all-MiniLM-L6-v2') -> 'ChunkRetriever':
        "
        Load a previously saved database.
        
        Args:
            directory: Directory containing the saved database files
            task_domain: Domain of the task
            model_name: Name of the sentence transformer model to use
            
        Returns:
            ChunkRetriever instance with loaded data
        "
        # Load metadata
        with open(os.path.join(directory, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
            
        # Create instance
        instance = cls(task_domain, model_name=model_name, random_seed=metadata['random_seed'])
        instance.chunks = metadata['chunks']
        
        # Load FAISS index
        instance.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
        
        # Make sure the index is properly loaded
        if instance.index is None:
            # Rebuild index if loading failed
            instance._build_index()
        
        return instance


class RerankingRetriever(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, rerank_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.base_retriever = base_retriever
        self.rerank_model = CrossEncoder(rerank_model_name)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        # Get initial results
        initial_results = self.base_retriever.retrieve(query, k=k*2)  # Retrieve more initially

        # Check if we have any results
        if not initial_results:
            print(f"Warning: No results found for query: {query}")
            return []

        # Prepare pairs for re-ranking
        pairs = [[query, doc] for doc, _ in initial_results]

        # Re-rank
        scores = self.rerank_model.predict(pairs)

        # Sort and return top k
        reranked_results = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)[:k]
        return [(doc, score) for (doc, _), score in reranked_results]



Your task is:
1. Analyse the retrieval evaluation metrics and my current code base
2. Think how we can implement the retrieval evaluation metrics without running the whole RAG system
3. Implement it in a new python script
"""
# "question": "An energy company is planning to expand its natural gas processing capabilities in the Panhandle region. Given the information about the existing infrastructure, which strategy would most effectively increase the company's ability to handle varying ethane market conditions while maximizing operational flexibility?\n",

# "choices": [
#       "A) Construct a new processing plant with 200 MMcfd inlet capacity, connected to multiple pipeline systems",
#       "B) Upgrade all existing plants to allow for 24-hour unattended operation",
#       "C) Build additional field compressor stations to increase gathering capacity from producers",
#       "D) Implement a system to dynamically switch between ethane recovery and rejection modes across all plants"
#     ],

# Your task is answer the following:
# 1. What's the correct answer?
# 2. Reasoning
# 3. Based on the Bloom’s taxonomy, which level of question is this?
# 4. Do you need some supporting documents to solve this question?
# 4. How easy is to solve this question with documentation?

# Could you make a multiple choice question and answer with 1 correct and 3 distractors.
# The question must be very difficult and aim for L3 or 4 of Bloom's taxonomy. 
# Also the answer is should very difficult to derive without those documentations.

# "documentation": [
#       "During 2012, we completed construction of and placed into service one of the four processing facilities. Phase I expansion of the facility was completed in March bringing inlet capacity to 80 MMcfd. Phase II expansion of the same facility was completed in June bringing the total inlet capacity of the plant to 140 MMcfd. This addition to the Panhandle System enables us to meet our current and expected future processing requirements in this area. We are also improving the connectivity between plants to enable us to better utilize our Panhandle processing capabilities and better serve the growing needs of the area producers, including those in the Granite Wash.\nAll four plants are capable of operating in high ethane recovery mode or in ethane rejection mode and have instrumentation allowing for unattended operation of up to 16 hours per day.\nThe Panhandle System is comprised of a number of pipeline gathering systems and 43 field compressor stations that gather natural gas, directly or indirectly, to the plants. These gathering systems are located in Beaver, Ellis, Harper, and Roger Mills Counties in Oklahoma and Hansford, Hemphill, Hutchinson, Lipscomb, Ochiltree, Roberts and Wheeler Counties in Texas.\nNatural Gas Supply and Markets for Sale of Natural Gas and NGLs. The residue gas from the Antelope Hills plant is delivered into Southern Star Central Gas or Northern Natural Gas pipelines for sale or transportation to market. The NGLs produced at the Antelope Hills plant are delivered into ONEOK Hydrocarbon’s pipeline system for transportation to and fractionation at ONEOK’s Conway fractionator.\nThe residue gas from the Beaver plant is delivered into Northern Natural Gas, Southern Star Central Gas or ANR Pipeline Company pipelines for sale or transportation to market. The NGLs produced at the Beaver plant are delivered into ONEOK Hydrocarbon’s pipeline system for transportation to and fractionation at ONEOK’s Conway fractionator.\nThe residue gas from the Spearman plant is delivered into Northern Natural Gas or ANR pipelines for sale or transportation to market. The NGLs produced at the Spearman plant are delivered into MAPCO’s (Mid-America Pipeline Company) pipeline system. MAPCO’s pipeline system has the flexibility of delivering the NGLs to either Mont Belvieu or Conway for fractionation.\nThe residue gas from the Sweetwater plant is delivered into Oklahoma Gas Transmission or ANR pipelines for sale or transportation to market. The NGLs produced at the Sweetwater plant are delivered into ONEOK Hydrocarbon’s pipeline system for transportation and fractionation, with the majority being handled at ONEOK’s Conway fractionator and a portion being delivered to the Mont Belvieu markets.\nCrescent System\nGeneral. The Crescent System is a natural gas gathering system stretching over seven counties within central Oklahoma’s Sooner Trend. The system consists of approximately 1,724 miles of natural gas gathering pipelines, ranging in size from two to 10 inches in diameter, and the Crescent natural gas processing plant located in Logan County, Oklahoma. Fourteen compressor stations are operating across the Crescent System. We continue to look at potential growth opportunities to service the Mississippian Lime formation.\nThe Crescent plant is a NGL recovery plant with current capacity of approximately 40 MMcfd. The Crescent facility also includes a gas engine-driven generator which is routinely operated, making the plant self-sufficient with respect to electric power. The cost of fuel (residue gas) for the generator is borne by the producers under the terms of their respective gas contracts.",
#       "The Spearman plant has 100 MMcfd of inlet capacity. The plant is capable of operating in high ethane recovery mode or in ethane rejection mode and has instrumentation allowing for unattended operation of up to 16 hours per day.\nThe Sweetwater plant is capable of operating in high ethane recovery mode or in ethane rejection mode and has instrumentation allowing for unattended operation of up to 16 hours per day.\nIn conjunction with the acquisition of the Sweetwater plant, two new gas compressor stations were installed; one is located on the east end of the North Canadian pipeline and the other on the east end of the Hemphill pipeline.\nNatural Gas Supply and Markets for Sale of Natural Gas and NGLs. The supply in the Panhandle System comes from approximately 203 producers pursuant to 332 contracts. The residue gas from the Beaver plant can be delivered into the Northern Natural Gas, Southern Star Central Gas or ANR Pipeline Company pipelines for sale or transportation to market. The NGLs produced at the Beaver plant are delivered into ONEOK Hydrocarbon’s pipeline system for transportation to and fractionation at ONEOK’s Conway fractionator.\nThe residue gas from the Spearman plant is delivered into Northern Natural Gas pipelines for sale or transportation to market. The NGLs produced at the Spearman plant are delivered into MAPCO’s (Mid-America Pipeline Company) pipeline system. MAPCO’s pipeline system has the flexibility of delivering the NGLs to either Mont Belvieu or Conway for fractionation.\nThe residue gas from the Sweetwater plant is delivered into Northern Natural Gas pipelines for sale or transportation to market. The NGLs produced at the Sweetwater plant are delivered into ONEOK Hydrocarbon’s pipeline system for transportation to and fractionation at ONEOK’s Conway fractionator.\nCrossroads System General. The Crossroads System is a natural gas gathering system located in the southeast portion of Harrison County, Texas. The Crossroads System consists of approximately eight miles of natural gas gathering pipelines, ranging in size from eight to twelve inches in diameter, and the Crossroads plant. The Crossroads System also includes approximately 20 miles of six-inch NGL pipeline that transport the NGLs produced at the Crossroads plant to the Panola Pipeline.\nThe Crossroads plant has 80 MMcfd of inlet capacity. The plant is capable of operating in high ethane recovery mode or in ethane rejection mode and has instrumentation allowing for unattended operation of up to 16 hours per day.\nNatural Gas Supply and Markets for Sale of Natural Gas and NGLs. The natural gas on the Crossroads System originates from the Bethany Field from where we have contracted with five producers. The Crossroads System delivers the residue gas from the Crossroads plant into the CenterPoint Energy pipeline for sale or transportation to market. The NGLs produced at the Crossroads plant are delivered into the Panola Pipeline for transportation to Mont Belvieu, Texas for fractionation.\nCrescent System General. The Crescent System is a natural gas gathering system stretching over seven counties within central Oklahoma’s Sooner Trend. The system consists of approximately 1,701 miles of natural gas gathering pipelines, ranging in size from two to 10 inches in diameter, and the Crescent natural gas processing plant located in Logan County, Oklahoma. Fifteen compressor stations are operating across the Crescent System.\nThe Crescent plant is a NGL recovery plant with current capacity of approximately 40 MMcfd. The Crescent facility also includes a gas engine-driven generator which is routinely operated, making the plant self-sufficient with respect to electric power. The cost of fuel (residue gas) for the generator is borne by the producers under the terms of their respective gas contracts."
#     ]

print(model.invoke(prompt))