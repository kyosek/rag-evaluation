from LLMServer.gcp.claude_instant import ClaudeGcp

model = ClaudeGcp()

prompt = """I am developing a python script to perform RAG. I've already created contexutualised embeddings database.
My current script looks like this.

Current RAG script:

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

from MultiHopData.generate_exam import ChunkRetriever, Chunk
from LLMServer.llama.llama_instant import LlamaModel
from LLMServer.llama_gcp.llama_gcp_instant import LlamaGcpModel
from LLMServer.gcp.claude_instant import ClaudeGcp

nltk.download('punkt_tab')


@dataclass
class ExamQuestion:
    question: str
    choices: List[str]
    correct_answer: str
    documentation: List[str]


class BaseRetriever(ABC):
    "Abstract base class for different retrieval methods."
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        "Retrieve relevant documents for a query."
        pass


class FAISSRetriever(BaseRetriever):
    "Dense retrieval using FAISS."
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
    "Sparse retrieval using BM25."
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
    "Combines multiple retrievers with optional weights."
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
        "Load exam questions from JSON file."
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
        "Solve a single exam question using RAG with LLM."
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question.question, k=self.n_documents)
        
        # Prepare context
        context = "\n".join([doc for doc, _ in retrieved_docs])
        
        # Use LLM to generate answer
        formatted_choices = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(question.choices))
    
        # Construct a more structured prompt with system and user roles
        prompt = f"[INST] <<SYS>>
        You are an AI assistant taking a multiple choice exam. Your task is to:
        1. Read the given question and supporting document carefully
        2. Analyze the choices
        3. Select the most appropriate answer
        4. Respond with ONLY the letter (A, B, C, or D) of the correct answer
        <</SYS>>

        Question: {question.question}

        Choices:
        {formatted_choices}
        
        Supporting document:
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

        Your answer (one letter only): [/INST]"

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
    
    def evaluate_performance(self, questions: List[ExamQuestion], model) -> Dict[str, float]:
        "Evaluate the solver's performance on a set of questions."
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


def main(task_domain: str, retriever_type: str, model_name: str):
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
        solver = ExamSolver(faiss_retriever)
    elif retriever_type == "Sparse":
        solver = ExamSolver(bm25_retriever)
    elif retriever_type == "Hybrid":
        solver = ExamSolver(hybrid_retriever)
    
    # Load and solve exam
    if model_name == "GCP":
        print("Using transformer")
    elif model_name == "claude":
        model = ClaudeGcp()
    else:
        print("Using Llama-cpp")
        # model = LlamaModel(model_path=model_path)
        
    questions = solver.load_exam(f"MultiHopData/{task_domain}/exam_cleaned_1000_42.json")
    metrics = solver.evaluate_performance(questions, model)
    
    print(f"Exam Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")


if __name__ == "__main__":
    task_domain = "SecFilings"
    # retriever_type = "Dense"
    # retriever_type = "Sparse"
    retriever_type = "Hybrid"
    model_name = "claude"
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    
    for task_domain in task_domains:
        print(f"Processing {task_domain}")
        main(task_domain, retriever_type, model_name)
        
class ChunkRetriever:
    def __init__(self, task_domain: str, model_name: str = 'all-MiniLM-L6-v2', random_seed: Optional[int] = None):
        "
        Initialize the chunk retriever with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            random_seed: Seed for random operations (optional)
        "
        self.model = SentenceTransformer(model_name)
        self.task_domain = task_domain
        self.index = None
        self.chunks: List[Chunk] = []
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
        
    def load_documents(self, json_file: str) -> None:
        "
        Load documents from JSON file and store chunks.
        
        Args:
            json_file: Path to the JSON file containing documents
        "
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
        "Build FAISS index from chunks."
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
        "
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
        
        
class ContextualChunkRetriever:
    def __init__(self, task_domain: str, model_name: str = 'all-MiniLM-L6-v2', random_seed: Optional[int] = None):
        self.model = SentenceTransformer(model_name)
        self.task_domain = task_domain
        self.index = None
        self.chunks: List[Chunk] = []
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
        
        self.claude = ClaudeGcp(model_name="claude-3-5-haiku@20241022")
        self.gemini = GeminiGcp(model_name="gemini-1.5-flash-002")

    def generate_context(self, doc_content: str, chunk_content: str, model_name) -> str:
        prompt = f'''
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
        '''
        llm = model_name

        response = llm.invoke(prompt)
        return response.strip()

    def load_documents(self, json_file: str) -> None:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for doc in tqdm(data, desc="Processing documents"):
            doc_id = doc['doc_id']
            doc_content = "\n".join([chunk['content'] for chunk in doc['chunks']])
            
            for chunk in doc['chunks']:
                try:
                    context = self.generate_context(doc_content, chunk['content'], self.gemini)
                except:
                    print("Generating with Claude")
                    context = self.generate_context(doc_content, chunk['content'], self.claude)
                chunk_obj = Chunk(
                    chunk_id=chunk['chunk_id'],
                    doc_id=doc_id,
                    content=chunk['content'],
                    original_index=chunk['original_index'],
                    context=context
                )
                self.chunks.append(chunk_obj)
        
        self._build_index()

    def _build_index(self) -> None:
        embeddings = self.model.encode([f"{chunk.content}\n\nContext: {chunk.context}" for chunk in self.chunks])
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def sample_chunks(self, n: int, seed: Optional[int] = None) -> List[Chunk]:
        if seed is not None:
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
    ) -> List[tuple[Chunk, float]]:
        query_embedding = self.model.encode([f"{query_chunk.content}\n\nContext: {query_chunk.context}"])
        faiss.normalize_L2(query_embedding)
        
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
        
        metadata = {
            'chunks': self.chunks,
            'random_seed': self.random_seed
        }
        with open(os.path.join(directory, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)

    @classmethod
    def load_database(cls, directory: str, task_domain: str, model_name: str = 'all-MiniLM-L6-v2') -> 'ContextualChunkRetriever':
        with open(os.path.join(directory, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
        
        instance = cls(task_domain, model_name=model_name, random_seed=metadata['random_seed'])
        instance.chunks = metadata['chunks']
        
        instance.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
        
        if instance.index is None:
            instance._build_index()
        
        return instance

def main(task_domain: str):
    chunk_retriever = ContextualChunkRetriever(task_domain, random_seed=42)
    chunk_retriever.load_documents(f"MultiHopData/{task_domain}/docs_chunk.json")
    chunk_retriever.save_database(f"MultiHopData/{task_domain}/contextual_chunk_flash_database")
"

`ChunkRetriever` is used to to create the database and retrieve chunks.

I saw this notebook and would like to adapt this to my script (will write a new script for contextual embedding).

Notebook:
```
{
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Contextual Embeddings\n",
    "\n",
    "With basic RAG, each embedded chunk contains a potentially useful piece of information, but these chunks lack context. With Contextual Embeddings, we create a variation on the embedding itself by adding more context to each text chunk before embedding it. Specifically, we use Claude to create a concise context that explains the chunk using the context of the overall document. In the case of our codebases dataset, we can provide both the chunk and the full file that each chunk was found within to an LLM, then produce the context. Then, we will combine this 'context' and the raw text chunk together into a single text block prior to creating each embedding.\n",
    "\n",
    "### Additional Considerations: Cost and Latency\n",
    "\n",
    "The extra work we're doing to 'situate' each document happens only at ingestion time: it's a cost you'll pay once when you store each document (and periodically in the future if you have a knowledge base that updates over time). There are many approaches like HyDE (hypothetical document embeddings) which involve performing steps to improve the representation of the query prior to executing a search. These techniques have shown to be moderately effective, but they add significant latency at runtime.\n",
    "\n",
    "[Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) also makes this much more cost effective. Creating contextual embeddings requires us to pass the same document to the model for every chunk we want to generate extra context for. With prompt caching, we can write the overall doc to the cache once, and then because we're doing our ingestion job all in sequence, we can just read the document from cache as we generate context for each chunk within that document (the information you write to the cache has a 5 minute time to live). This means that the first time we pass a document to the model, we pay a bit more to write it to the cache, but for each subsequent API call that contains that doc, we receive  a 90% discount on all of the input tokens read from the cache. Assuming 800 token chunks, 8k token documents, 50 token context instructions, and 100 tokens of context per chunk, the cost to generate contextualized chunks is $1.02 per million document tokens.\n",
    "\n",
    "When you load data into your ContextualVectorDB below, you'll see in logs just how big this impact is. \n",
    "\n",
    "Warning: some smaller embedding models have a fixed input token limit. Contextualizing the chunk makes it longer, so if you notice much worse performance from contextualized embeddings, the contextualized chunk is likely getting truncated"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Situated context: This chunk describes the `DiffExecutor` struct, which is an executor for differential fuzzing. It wraps two executors that are run sequentially with the same input, and also runs the secondary executor in the `run_target` method.\n",
      "Input tokens: 366\n",
      "Output tokens: 55\n",
      "Cache creation input tokens: 3046\n",
      "Cache read input tokens: 0\n"
     ]
    }
   ],
   "source": [
    "DOCUMENT_CONTEXT_PROMPT = \"\"\"\n",
    "<document>\n",
    "{doc_content}\n",
    "</document>\n",
    "\"\"\"\n",
    "\n",
    "CHUNK_CONTEXT_PROMPT = \"\"\"\n",
    "Here is the chunk we want to situate within the whole document\n",
    "<chunk>\n",
    "{chunk_content}\n",
    "</chunk>\n",
    "\n",
    "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.\n",
    "Answer only with the succinct context and nothing else.\n",
    "\"\"\"\n",
    "\n",
    "def situate_context(doc: str, chunk: str) -> str:\n",
    "    response = client.beta.prompt_caching.messages.create(\n",
    "        model=\"claude-3-haiku-20240307\",\n",
    "        max_tokens=1024,\n",
    "        temperature=0.0,\n",
    "        messages=[\n",
    "            {\n",
    "                \"role\": \"user\", \n",
    "                \"content\": [\n",
    "                    {\n",
    "                        \"type\": \"text\",\n",
    "                        \"text\": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),\n",
    "                        \"cache_control\": {\"type\": \"ephemeral\"} #we will make use of prompt caching for the full documents\n",
    "                    },\n",
    "                    {\n",
    "                        \"type\": \"text\",\n",
    "                        \"text\": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),\n",
    "                    }\n",
    "                ]\n",
    "            }\n",
    "        ],\n",
    "        extra_headers={\"anthropic-beta\": \"prompt-caching-2024-07-31\"}\n",
    "    )\n",
    "    return response\n",
    "\n",
    "jsonl_data = load_jsonl('data/evaluation_set.jsonl')\n",
    "# Example usage\n",
    "doc_content = jsonl_data[0]['golden_documents'][0]['content']\n",
    "chunk_content = jsonl_data[0]['golden_chunks'][0]['content']\n",
    "\n",
    "response = situate_context(doc_content, chunk_content)\n",
    "print(f\"Situated context: {response.content[0].text}\")\n",
    "\n",
    "# Print cache performance metrics\n",
    "print(f\"Input tokens: {response.usage.input_tokens}\")\n",
    "print(f\"Output tokens: {response.usage.output_tokens}\")\n",
    "print(f\"Cache creation input tokens: {response.usage.cache_creation_input_tokens}\")\n",
    "print(f\"Cache read input tokens: {response.usage.cache_read_input_tokens}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 318,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pickle\n",
    "import json\n",
    "import numpy as np\n",
    "import voyageai\n",
    "from typing import List, Dict, Any\n",
    "from tqdm import tqdm\n",
    "import anthropic\n",
    "import threading\n",
    "import time\n",
    "from concurrent.futures import ThreadPoolExecutor, as_completed\n",
    "\n",
    "class ContextualVectorDB:\n",
    "    def __init__(self, name: str, voyage_api_key=None, anthropic_api_key=None):\n",
    "        if voyage_api_key is None:\n",
    "            voyage_api_key = os.getenv(\"VOYAGE_API_KEY\")\n",
    "        if anthropic_api_key is None:\n",
    "            anthropic_api_key = os.getenv(\"ANTHROPIC_API_KEY\")\n",
    "        \n",
    "        self.voyage_client = voyageai.Client(api_key=voyage_api_key)\n",
    "        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)\n",
    "        self.name = name\n",
    "        self.embeddings = []\n",
    "        self.metadata = []\n",
    "        self.query_cache = {}\n",
    "        self.db_path = f\"./data/{name}/contextual_vector_db.pkl\"\n",
    "\n",
    "        self.token_counts = {\n",
    "            'input': 0,\n",
    "            'output': 0,\n",
    "            'cache_read': 0,\n",
    "            'cache_creation': 0\n",
    "        }\n",
    "        self.token_lock = threading.Lock()\n",
    "\n",
    "    def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:\n",
    "        DOCUMENT_CONTEXT_PROMPT = \"\"\"\n",
    "        <document>\n",
    "        {doc_content}\n",
    "        </document>\n",
    "        \"\"\"\n",
    "\n",
    "        CHUNK_CONTEXT_PROMPT = \"\"\"\n",
    "        Here is the chunk we want to situate within the whole document\n",
    "        <chunk>\n",
    "        {chunk_content}\n",
    "        </chunk>\n",
    "\n",
    "        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.\n",
    "        Answer only with the succinct context and nothing else.\n",
    "        \"\"\"\n",
    "\n",
    "        response = self.anthropic_client.beta.prompt_caching.messages.create(\n",
    "            model=\"claude-3-haiku-20240307\",\n",
    "            max_tokens=1000,\n",
    "            temperature=0.0,\n",
    "            messages=[\n",
    "                {\n",
    "                    \"role\": \"user\", \n",
    "                    \"content\": [\n",
    "                        {\n",
    "                            \"type\": \"text\",\n",
    "                            \"text\": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),\n",
    "                            \"cache_control\": {\"type\": \"ephemeral\"} #we will make use of prompt caching for the full documents\n",
    "                        },\n",
    "                        {\n",
    "                            \"type\": \"text\",\n",
    "                            \"text\": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),\n",
    "                        },\n",
    "                    ]\n",
    "                },\n",
    "            ],\n",
    "            extra_headers={\"anthropic-beta\": \"prompt-caching-2024-07-31\"}\n",
    "        )\n",
    "        return response.content[0].text, response.usage\n",
    "\n",
    "    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 1):\n",
    "        if self.embeddings and self.metadata:\n",
    "            print(\"Vector database is already loaded. Skipping data loading.\")\n",
    "            return\n",
    "        if os.path.exists(self.db_path):\n",
    "            print(\"Loading vector database from disk.\")\n",
    "            self.load_db()\n",
    "            return\n",
    "\n",
    "        texts_to_embed = []\n",
    "        metadata = []\n",
    "        total_chunks = sum(len(doc['chunks']) for doc in dataset)\n",
    "\n",
    "        def process_chunk(doc, chunk):\n",
    "            #for each chunk, produce the context\n",
    "            contextualized_text, usage = self.situate_context(doc['content'], chunk['content'])\n",
    "            with self.token_lock:\n",
    "                self.token_counts['input'] += usage.input_tokens\n",
    "                self.token_counts['output'] += usage.output_tokens\n",
    "                self.token_counts['cache_read'] += usage.cache_read_input_tokens\n",
    "                self.token_counts['cache_creation'] += usage.cache_creation_input_tokens\n",
    "            \n",
    "            return {\n",
    "                #append the context to the original text chunk\n",
    "                'text_to_embed': f\"{chunk['content']}\\n\\n{contextualized_text}\",\n",
    "                'metadata': {\n",
    "                    'doc_id': doc['doc_id'],\n",
    "                    'original_uuid': doc['original_uuid'],\n",
    "                    'chunk_id': chunk['chunk_id'],\n",
    "                    'original_index': chunk['original_index'],\n",
    "                    'original_content': chunk['content'],\n",
    "                    'contextualized_content': contextualized_text\n",
    "                }\n",
    "            }\n",
    "\n",
    "        print(f\"Processing {total_chunks} chunks with {parallel_threads} threads\")\n",
    "        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:\n",
    "            futures = []\n",
    "            for doc in dataset:\n",
    "                for chunk in doc['chunks']:\n",
    "                    futures.append(executor.submit(process_chunk, doc, chunk))\n",
    "            \n",
    "            for future in tqdm(as_completed(futures), total=total_chunks, desc=\"Processing chunks\"):\n",
    "                result = future.result()\n",
    "                texts_to_embed.append(result['text_to_embed'])\n",
    "                metadata.append(result['metadata'])\n",
    "\n",
    "        self._embed_and_store(texts_to_embed, metadata)\n",
    "        self.save_db()\n",
    "\n",
    "        #logging token usage\n",
    "        print(f\"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}\")\n",
    "        print(f\"Total input tokens without caching: {self.token_counts['input']}\")\n",
    "        print(f\"Total output tokens: {self.token_counts['output']}\")\n",
    "        print(f\"Total input tokens written to cache: {self.token_counts['cache_creation']}\")\n",
    "        print(f\"Total input tokens read from cache: {self.token_counts['cache_read']}\")\n",
    "        \n",
    "        total_tokens = self.token_counts['input'] + self.token_counts['cache_read'] + self.token_counts['cache_creation']\n",
    "        savings_percentage = (self.token_counts['cache_read'] / total_tokens) * 100 if total_tokens > 0 else 0\n",
    "        print(f\"Total input token savings from prompt caching: {savings_percentage:.2f}% of all input tokens used were read from cache.\")\n",
    "        print(\"Tokens read from cache come at a 90 percent discount!\")\n",
    "\n",
    "    #we use voyage AI here for embeddings. Read more here: https://docs.voyageai.com/docs/embeddings\n",
    "    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):\n",
    "        batch_size = 128\n",
    "        result = [\n",
    "            self.voyage_client.embed(\n",
    "                texts[i : i + batch_size],\n",
    "                model=\"voyage-2\"\n",
    "            ).embeddings\n",
    "            for i in range(0, len(texts), batch_size)\n",
    "        ]\n",
    "        self.embeddings = [embedding for batch in result for embedding in batch]\n",
    "        self.metadata = data\n",
    "\n",
    "    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:\n",
    "        if query in self.query_cache:\n",
    "            query_embedding = self.query_cache[query]\n",
    "        else:\n",
    "            query_embedding = self.voyage_client.embed([query], model=\"voyage-2\").embeddings[0]\n",
    "            self.query_cache[query] = query_embedding\n",
    "\n",
    "        if not self.embeddings:\n",
    "            raise ValueError(\"No data loaded in the vector database.\")\n",
    "\n",
    "        similarities = np.dot(self.embeddings, query_embedding)\n",
    "        top_indices = np.argsort(similarities)[::-1][:k]\n",
    "        \n",
    "        top_results = []\n",
    "        for idx in top_indices:\n",
    "            result = {\n",
    "                \"metadata\": self.metadata[idx],\n",
    "                \"similarity\": float(similarities[idx]),\n",
    "            }\n",
    "            top_results.append(result)\n",
    "        return top_results\n",
    "\n",
    "    def save_db(self):\n",
    "        data = {\n",
    "            \"embeddings\": self.embeddings,\n",
    "            \"metadata\": self.metadata,\n",
    "            \"query_cache\": json.dumps(self.query_cache),\n",
    "        }\n",
    "        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)\n",
    "        with open(self.db_path, \"wb\") as file:\n",
    "            pickle.dump(data, file)\n",
    "\n",
    "    def load_db(self):\n",
    "        if not os.path.exists(self.db_path):\n",
    "            raise ValueError(\"Vector database file not found. Use load_data to create a new database.\")\n",
    "        with open(self.db_path, \"rb\") as file:\n",
    "            data = pickle.load(file)\n",
    "        self.embeddings = data[\"embeddings\"]\n",
    "        self.metadata = data[\"metadata\"]\n",
    "        self.query_cache = json.loads(data[\"query_cache\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 319,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing 737 chunks with 5 threads\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing chunks: 100%|██████████| 737/737 [02:37<00:00,  4.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Contextual Vector database loaded and saved. Total chunks processed: 737\n",
      "Total input tokens without caching: 500383\n",
      "Total output tokens: 40318\n",
      "Total input tokens written to cache: 341422\n",
      "Total input tokens read from cache: 2825073\n",
      "Total input token savings from prompt caching: 77.04% of all input tokens used were read from cache.\n",
      "Tokens read from cache come at a 90 percent discount!\n"
     ]
    }
   ],
   "source": [
    "# Load the transformed dataset\n",
    "with open('data/codebase_chunks.json', 'r') as f:\n",
    "    transformed_dataset = json.load(f)\n",
    "\n",
    "# Initialize the ContextualVectorDB\n",
    "contextual_db = ContextualVectorDB(\"my_contextual_db\")\n",
    "\n",
    "# Load and process the data\n",
    "#note: consider increasing the number of parallel threads to run this faster, or reducing the number of parallel threads if concerned about hitting your API rate limit\n",
    "contextual_db.load_data(transformed_dataset, parallel_threads=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 360,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating retrieval: 100%|██████████| 248/248 [00:06<00:00, 39.53it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pass@5: 86.37%\n",
      "Total Score: 0.8637192780337941\n",
      "Total queries: 248\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating retrieval: 100%|██████████| 248/248 [00:06<00:00, 40.05it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pass@10: 92.81%\n",
      "Total Score: 0.9280913978494625\n",
      "Total queries: 248\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating retrieval: 100%|██████████| 248/248 [00:06<00:00, 39.64it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pass@20: 93.78%\n",
      "Total Score: 0.9378360215053763\n",
      "Total queries: 248\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "r5 = evaluate_db(contextual_db, 'data/evaluation_set.jsonl', 5)\n",
    "r10 = evaluate_db(contextual_db, 'data/evaluation_set.jsonl', 10)\n",
    "r20 = evaluate_db(contextual_db, 'data/evaluation_set.jsonl', 20)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Contextual BM25\n",
    "\n",
    "Contextual embeddings is an improvement on traditional semantic search RAG, but we can improve performance further. In this section we'll show you how you can use contextual embeddings and *contextual* BM25 together. While you can see performance gains by pairing these techniques together without the context, adding context to these methods reduces the top-20-chunk retrieval failure rate by 42%.\n",
    "\n",
    "BM25 is a probabilistic ranking function that improves upon TF-IDF. It scores documents based on query term frequency, while accounting for document length and term saturation. BM25 is widely used in modern search engines for its effectiveness in ranking relevant documents. For more details, see [this blog post](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables). We'll use elastic search for the BM25 portion of this section, which will require you to have the elasticsearch library installed and it will also require you to spin up an Elasticsearch server in the background. The easiest way to do this is to install [docker](https://docs.docker.com/engine/install/) and run the following docker command:\n",
    "\n",
    "`docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e \"discovery.type=single-node\" -e \"xpack.security.enabled=false\" elasticsearch:8.8.0`\n",
    "\n",
    "One difference between a typical BM25 search and what we'll do in this section is that, for each chunk, we'll run each BM25 search on both the chunk content and the additional context that we generated in the previous section. From there, we'll use a technique called reciprocal rank fusion to merge the results from our BM25 search with our semantic search results. This allows us to perform a hybrid search across both our BM25 corpus and vector DB to return the most optimal documents for a given query.\n",
    "\n",
    "In the function below, we allow you the option to add weightings to the semantic search and BM25 search documents as you merge them with Reciprocal Rank Fusion. By default, we set these to 0.8 for the semantic search results and 0.2 to the BM25 results. We'd encourage you to experiment with different values here."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 369,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "from typing import List, Dict, Any\n",
    "from tqdm import tqdm\n",
    "from elasticsearch import Elasticsearch\n",
    "from elasticsearch.helpers import bulk\n",
    "\n",
    "class ElasticsearchBM25:\n",
    "    def __init__(self, index_name: str = \"contextual_bm25_index\"):\n",
    "        self.es_client = Elasticsearch(\"http://localhost:9200\")\n",
    "        self.index_name = index_name\n",
    "        self.create_index()\n",
    "\n",
    "    def create_index(self):\n",
    "        index_settings = {\n",
    "            \"settings\": {\n",
    "                \"analysis\": {\"analyzer\": {\"default\": {\"type\": \"english\"}}},\n",
    "                \"similarity\": {\"default\": {\"type\": \"BM25\"}},\n",
    "                \"index.queries.cache.enabled\": False  # Disable query cache\n",
    "            },\n",
    "            \"mappings\": {\n",
    "                \"properties\": {\n",
    "                    \"content\": {\"type\": \"text\", \"analyzer\": \"english\"},\n",
    "                    \"contextualized_content\": {\"type\": \"text\", \"analyzer\": \"english\"},\n",
    "                    \"doc_id\": {\"type\": \"keyword\", \"index\": False},\n",
    "                    \"chunk_id\": {\"type\": \"keyword\", \"index\": False},\n",
    "                    \"original_index\": {\"type\": \"integer\", \"index\": False},\n",
    "                }\n",
    "            },\n",
    "        }\n",
    "        if not self.es_client.indices.exists(index=self.index_name):\n",
    "            self.es_client.indices.create(index=self.index_name, body=index_settings)\n",
    "            print(f\"Created index: {self.index_name}\")\n",
    "\n",
    "    def index_documents(self, documents: List[Dict[str, Any]]):\n",
    "        actions = [\n",
    "            {\n",
    "                \"_index\": self.index_name,\n",
    "                \"_source\": {\n",
    "                    \"content\": doc[\"original_content\"],\n",
    "                    \"contextualized_content\": doc[\"contextualized_content\"],\n",
    "                    \"doc_id\": doc[\"doc_id\"],\n",
    "                    \"chunk_id\": doc[\"chunk_id\"],\n",
    "                    \"original_index\": doc[\"original_index\"],\n",
    "                },\n",
    "            }\n",
    "            for doc in documents\n",
    "        ]\n",
    "        success, _ = bulk(self.es_client, actions)\n",
    "        self.es_client.indices.refresh(index=self.index_name)\n",
    "        return success\n",
    "\n",
    "    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:\n",
    "        self.es_client.indices.refresh(index=self.index_name)  # Force refresh before each search\n",
    "        search_body = {\n",
    "            \"query\": {\n",
    "                \"multi_match\": {\n",
    "                    \"query\": query,\n",
    "                    \"fields\": [\"content\", \"contextualized_content\"],\n",
    "                }\n",
    "            },\n",
    "            \"size\": k,\n",
    "        }\n",
    "        response = self.es_client.search(index=self.index_name, body=search_body)\n",
    "        return [\n",
    "            {\n",
    "                \"doc_id\": hit[\"_source\"][\"doc_id\"],\n",
    "                \"original_index\": hit[\"_source\"][\"original_index\"],\n",
    "                \"content\": hit[\"_source\"][\"content\"],\n",
    "                \"contextualized_content\": hit[\"_source\"][\"contextualized_content\"],\n",
    "                \"score\": hit[\"_score\"],\n",
    "            }\n",
    "            for hit in response[\"hits\"][\"hits\"]\n",
    "        ]\n",
    "    \n",
    "def create_elasticsearch_bm25_index(db: ContextualVectorDB):\n",
    "    es_bm25 = ElasticsearchBM25()\n",
    "    es_bm25.index_documents(db.metadata)\n",
    "    return es_bm25\n",
    "\n",
    "def retrieve_advanced(query: str, db: ContextualVectorDB, es_bm25: ElasticsearchBM25, k: int, semantic_weight: float = 0.8, bm25_weight: float = 0.2):\n",
    "    num_chunks_to_recall = 150\n",
    "\n",
    "    # Semantic search\n",
    "    semantic_results = db.search(query, k=num_chunks_to_recall)\n",
    "    ranked_chunk_ids = [(result['metadata']['doc_id'], result['metadata']['original_index']) for result in semantic_results]\n",
    "\n",
    "    # BM25 search using Elasticsearch\n",
    "    bm25_results = es_bm25.search(query, k=num_chunks_to_recall)\n",
    "    ranked_bm25_chunk_ids = [(result['doc_id'], result['original_index']) for result in bm25_results]\n",
    "\n",
    "    # Combine results\n",
    "    chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))\n",
    "    chunk_id_to_score = {}\n",
    "\n",
    "    # Initial scoring with weights\n",
    "    for chunk_id in chunk_ids:\n",
    "        score = 0\n",
    "        if chunk_id in ranked_chunk_ids:\n",
    "            index = ranked_chunk_ids.index(chunk_id)\n",
    "            score += semantic_weight * (1 / (index + 1))  # Weighted 1/n scoring for semantic\n",
    "        if chunk_id in ranked_bm25_chunk_ids:\n",
    "            index = ranked_bm25_chunk_ids.index(chunk_id)\n",
    "            score += bm25_weight * (1 / (index + 1))  # Weighted 1/n scoring for BM25\n",
    "        chunk_id_to_score[chunk_id] = score\n",
    "\n",
    "    # Sort chunk IDs by their scores in descending order\n",
    "    sorted_chunk_ids = sorted(\n",
    "        chunk_id_to_score.keys(), key=lambda x: (chunk_id_to_score[x], x[0], x[1]), reverse=True\n",
    "    )\n",
    "\n",
    "    # Assign new scores based on the sorted order\n",
    "    for index, chunk_id in enumerate(sorted_chunk_ids):\n",
    "        chunk_id_to_score[chunk_id] = 1 / (index + 1)\n",
    "\n",
    "    # Prepare the final results\n",
    "    final_results = []\n",
    "    semantic_count = 0\n",
    "    bm25_count = 0\n",
    "    for chunk_id in sorted_chunk_ids[:k]:\n",
    "        chunk_metadata = next(chunk for chunk in db.metadata if chunk['doc_id'] == chunk_id[0] and chunk['original_index'] == chunk_id[1])\n",
    "        is_from_semantic = chunk_id in ranked_chunk_ids\n",
    "        is_from_bm25 = chunk_id in ranked_bm25_chunk_ids\n",
    "        final_results.append({\n",
    "            'chunk': chunk_metadata,\n",
    "            'score': chunk_id_to_score[chunk_id],\n",
    "            'from_semantic': is_from_semantic,\n",
    "            'from_bm25': is_from_bm25\n",
    "        })\n",
    "        \n",
    "        if is_from_semantic and not is_from_bm25:\n",
    "            semantic_count += 1\n",
    "        elif is_from_bm25 and not is_from_semantic:\n",
    "            bm25_count += 1\n",
    "        else:  # it's in both\n",
    "            semantic_count += 0.5\n",
    "            bm25_count += 0.5\n",
    "\n",
    "    return final_results, semantic_count, bm25_count\n",
    "\n",
    "def load_jsonl(file_path: str) -> List[Dict[str, Any]]:\n",
    "    with open(file_path, 'r') as file:\n",
    "        return [json.loads(line) for line in file]\n",
    "\n",
    "def evaluate_db_advanced(db: ContextualVectorDB, original_jsonl_path: str, k: int):\n",
    "    original_data = load_jsonl(original_jsonl_path)\n",
    "    es_bm25 = create_elasticsearch_bm25_index(db)\n",
    "    \n",
    "    try:\n",
    "        # Warm-up queries\n",
    "        warm_up_queries = original_data[:10]\n",
    "        for query_item in warm_up_queries:\n",
    "            _ = retrieve_advanced(query_item['query'], db, es_bm25, k)\n",
    "        \n",
    "        total_score = 0\n",
    "        total_semantic_count = 0\n",
    "        total_bm25_count = 0\n",
    "        total_results = 0\n",
    "        \n",
    "        for query_item in tqdm(original_data, desc=\"Evaluating retrieval\"):\n",
    "            query = query_item['query']\n",
    "            golden_chunk_uuids = query_item['golden_chunk_uuids']\n",
    "            \n",
    "            golden_contents = []\n",
    "            for doc_uuid, chunk_index in golden_chunk_uuids:\n",
    "                golden_doc = next((doc for doc in query_item['golden_documents'] if doc['uuid'] == doc_uuid), None)\n",
    "                if golden_doc:\n",
    "                    golden_chunk = next((chunk for chunk in golden_doc['chunks'] if chunk['index'] == chunk_index), None)\n",
    "                    if golden_chunk:\n",
    "                        golden_contents.append(golden_chunk['content'].strip())\n",
    "            \n",
    "            if not golden_contents:\n",
    "                print(f\"Warning: No golden contents found for query: {query}\")\n",
    "                continue\n",
    "            \n",
    "            retrieved_docs, semantic_count, bm25_count = retrieve_advanced(query, db, es_bm25, k)\n",
    "            \n",
    "            chunks_found = 0\n",
    "            for golden_content in golden_contents:\n",
    "                for doc in retrieved_docs[:k]:\n",
    "                    retrieved_content = doc['chunk']['original_content'].strip()\n",
    "                    if retrieved_content == golden_content:\n",
    "                        chunks_found += 1\n",
    "                        break\n",
    "            \n",
    "            query_score = chunks_found / len(golden_contents)\n",
    "            total_score += query_score\n",
    "            \n",
    "            total_semantic_count += semantic_count\n",
    "            total_bm25_count += bm25_count\n",
    "            total_results += len(retrieved_docs)\n",
    "        \n",
    "        total_queries = len(original_data)\n",
    "        average_score = total_score / total_queries\n",
    "        pass_at_n = average_score * 100\n",
    "        \n",
    "        semantic_percentage = (total_semantic_count / total_results) * 100 if total_results > 0 else 0\n",
    "        bm25_percentage = (total_bm25_count / total_results) * 100 if total_results > 0 else 0\n",
    "        \n",
    "        results = {\n",
    "            \"pass_at_n\": pass_at_n,\n",
    "            \"average_score\": average_score,\n",
    "            \"total_queries\": total_queries\n",
    "        }\n",
    "        \n",
    "        print(f\"Pass@{k}: {pass_at_n:.2f}%\")\n",
    "        print(f\"Average Score: {average_score:.2f}\")\n",
    "        print(f\"Total queries: {total_queries}\")\n",
    "        print(f\"Percentage of results from semantic search: {semantic_percentage:.2f}%\")\n",
    "        print(f\"Percentage of results from BM25: {bm25_percentage:.2f}%\")\n",
    "        \n",
    "        return results, {\"semantic\": semantic_percentage, \"bm25\": bm25_percentage}\n",
    "    \n",
    "    finally:\n",
    "        # Delete the Elasticsearch index\n",
    "        if es_bm25.es_client.indices.exists(index=es_bm25.index_name):\n",
    "            es_bm25.es_client.indices.delete(index=es_bm25.index_name)\n",
    "            print(f\"Deleted Elasticsearch index: {es_bm25.index_name}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 370,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Created index: contextual_bm25_index\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating retrieval: 100%|██████████| 248/248 [00:08<00:00, 28.36it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pass@5: 86.43%\n",
      "Average Score: 0.86\n",
      "Total queries: 248\n",
      "Percentage of results from semantic search: 55.12%\n",
      "Percentage of results from BM25: 44.88%\n",
      "Deleted Elasticsearch index: contextual_bm25_index\n",
      "Created index: contextual_bm25_index\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating retrieval: 100%|██████████| 248/248 [00:08<00:00, 28.02it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pass@10: 93.21%\n",
      "Average Score: 0.93\n",
      "Total queries: 248\n",
      "Percentage of results from semantic search: 58.35%\n",
      "Percentage of results from BM25: 41.65%\n",
      "Deleted Elasticsearch index: contextual_bm25_index\n",
      "Created index: contextual_bm25_index\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating retrieval: 100%|██████████| 248/248 [00:08<00:00, 28.15it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pass@20: 94.99%\n",
      "Average Score: 0.95\n",
      "Total queries: 248\n",
      "Percentage of results from semantic search: 61.94%\n",
      "Percentage of results from BM25: 38.06%\n",
      "Deleted Elasticsearch index: contextual_bm25_index\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "results5 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 5)\n",
    "results10 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 10)\n",
    "results20 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 20)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Adding a Reranking Step\n",
    "\n",
    "If you want to improve performance further, we recommend adding a re-ranking step. When using a re-ranker, you can retrieve more documents initially from your vector store, then use your re-ranker to select a subset of these documents. One common technique is to use re-ranking as a way to implement high precision hybrid search. You can use a combination of semantic search and keyword based search in your initial retrieval step (as we have done earlier in this guide), then use a re-ranking step to choose only the k most relevant docs from a combined list of documents returned by your semantic search and keyword search systems.\n",
    "\n",
    "Below, we'll demonstrate only the re-ranking step (skipping the hybrid search technique for now). You'll see that we retrieve 10x the number of documents than the number of final k documents we want to retrieve, then use a re-ranking model from Cohere to select the 10 most relevant results from that list. Adding the re-ranking step delivers a modest additional gain in performance. In our case, Pass@10 improves from 92.81% --> 94.79%."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 378,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cohere\n",
    "from typing import List, Dict, Any, Callable\n",
    "import json\n",
    "from tqdm import tqdm\n",
    "\n",
    "def load_jsonl(file_path: str) -> List[Dict[str, Any]]:\n",
    "    with open(file_path, 'r') as file:\n",
    "        return [json.loads(line) for line in file]\n",
    "\n",
    "def chunk_to_content(chunk: Dict[str, Any]) -> str:\n",
    "    original_content = chunk['metadata']['original_content']\n",
    "    contextualized_content = chunk['metadata']['contextualized_content']\n",
    "    return f\"{original_content}\\n\\nContext: {contextualized_content}\" \n",
    "\n",
    "def retrieve_rerank(query: str, db, k: int) -> List[Dict[str, Any]]:\n",
    "    co = cohere.Client( os.getenv(\"COHERE_API_KEY\"))\n",
    "    \n",
    "    # Retrieve more results than we normally would\n",
    "    semantic_results = db.search(query, k=k*10)\n",
    "    \n",
    "    # Extract documents for reranking, using the contextualized content\n",
    "    documents = [chunk_to_content(res) for res in semantic_results]\n",
    "\n",
    "    response = co.rerank(\n",
    "        model=\"rerank-english-v3.0\",\n",
    "        query=query,\n",
    "        documents=documents,\n",
    "        top_n=k\n",
    "    )\n",
    "    time.sleep(0.1)\n",
    "    \n",
    "    final_results = []\n",
    "    for r in response.results:\n",
    "        original_result = semantic_results[r.index]\n",
    "        final_results.append({\n",
    "            \"chunk\": original_result['metadata'],\n",
    "            \"score\": r.relevance_score\n",
    "        })\n",
    "    \n",
    "    return final_results\n",
    "\n",
    "def evaluate_retrieval_rerank(queries: List[Dict[str, Any]], retrieval_function: Callable, db, k: int = 20) -> Dict[str, float]:\n",
    "    total_score = 0\n",
    "    total_queries = len(queries)\n",
    "    \n",
    "    for query_item in tqdm(queries, desc=\"Evaluating retrieval\"):\n",
    "        query = query_item['query']\n",
    "        golden_chunk_uuids = query_item['golden_chunk_uuids']\n",
    "        \n",
    "        golden_contents = []\n",
    "        for doc_uuid, chunk_index in golden_chunk_uuids:\n",
    "            golden_doc = next((doc for doc in query_item['golden_documents'] if doc['uuid'] == doc_uuid), None)\n",
    "            if golden_doc:\n",
    "                golden_chunk = next((chunk for chunk in golden_doc['chunks'] if chunk['index'] == chunk_index), None)\n",
    "                if golden_chunk:\n",
    "                    golden_contents.append(golden_chunk['content'].strip())\n",
    "        \n",
    "        if not golden_contents:\n",
    "            print(f\"Warning: No golden contents found for query: {query}\")\n",
    "            continue\n",
    "        \n",
    "        retrieved_docs = retrieval_function(query, db, k)\n",
    "        \n",
    "        chunks_found = 0\n",
    "        for golden_content in golden_contents:\n",
    "            for doc in retrieved_docs[:k]:\n",
    "                retrieved_content = doc['chunk']['original_content'].strip()\n",
    "                if retrieved_content == golden_content:\n",
    "                    chunks_found += 1\n",
    "                    break\n",
    "        \n",
    "        query_score = chunks_found / len(golden_contents)\n",
    "        total_score += query_score\n",
    "    \n",
    "    average_score = total_score / total_queries\n",
    "    pass_at_n = average_score * 100\n",
    "    return {\n",
    "        \"pass_at_n\": pass_at_n,\n",
    "        \"average_score\": average_score,\n",
    "        \"total_queries\": total_queries\n",
    "    }\n",
    "\n",
    "def evaluate_db_advanced(db, original_jsonl_path, k):\n",
    "    original_data = load_jsonl(original_jsonl_path)\n",
    "    \n",
    "    def retrieval_function(query, db, k):\n",
    "        return retrieve_rerank(query, db, k)\n",
    "    \n",
    "    results = evaluate_retrieval_rerank(original_data, retrieval_function, db, k)\n",
    "    print(f\"Pass@{k}: {results['pass_at_n']:.2f}%\")\n",
    "    print(f\"Average Score: {results['average_score']}\")\n",
    "    print(f\"Total queries: {results['total_queries']}\")\n",
    "    return results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 380,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating retrieval: 100%|██████████| 248/248 [01:22<00:00,  2.99it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pass@5: 91.24%\n",
      "Average Score: 0.912442396313364\n",
      "Total queries: 248\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating retrieval: 100%|██████████| 248/248 [01:34<00:00,  2.63it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pass@10: 94.79%\n",
      "Average Score: 0.9479166666666667\n",
      "Total queries: 248\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating retrieval: 100%|██████████| 248/248 [02:08<00:00,  1.93it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pass@20: 96.30%\n",
      "Average Score: 0.9630376344086022\n",
      "Total queries: 248\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "results5 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 5)\n",
    "results10 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 10)\n",
    "results20 = evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 20)"
   ]
  }
```

Your task is:
1. Analyse the contextual embedding method
2. Think how to run the RAG system with it in the context of the existing codebase and contextualised database
3. Explain what components are needed to implement it
4. Write a python script(s) to do so (if it would be better, write a new script)
"""


print(model.invoke(prompt))