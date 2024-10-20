import json
from tqdm import tqdm

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever  #, DPRRetriever, MultiQueryRetriever
from langchain.document_loaders import JSONLoader
from typing import Dict, List

from ExamGenerator.utils import read_jsonl
from LLMServer.llama.llama_instant import LlamaModel

LLM = LlamaModel()
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def extract_text(data: Dict) -> List[str]:
    return [data.get("text", "")]


def load_corpus(data_path: str):
    loader = JSONLoader(
        file_path=data_path,
        jq_schema='.[]',
        content_key="text",
        text_content=False,
        json_lines=True
    )
    return loader.load()


def load_exam(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def get_retriever(retriever_type):
    if retriever_type == "bm25":
        bm25_retriever = BM25Retriever.from_documents(texts)
        bm25_retriever.k = 1  # Number of documents to retrieve
        return bm25_retriever
    vectorstore = Chroma.from_documents(texts, EMBEDDINGS)
    if retriever_type == "dpr":
        dpr_retriever = DPRRetriever(
            retriever=vectorstore.as_retriever(),
            embeddings=EMBEDDINGS
        )
        return dpr_retriever
    elif retriever_type == "multiqa":
        multiqa_retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(), llm=LLM
        )
        return multiqa_retriever
    else:
        raise ValueError("Invalid retriever type")


def retrieve_context(question: str, retriever):
    docs = retriever.invoke(question)

    retrieved_docs_text = [
        doc.page_content for doc in docs
    ]
    context = "\nRetrieved contexts:\n"
    context += "".join(
        [f"Context {i}:::\n{doc}" for i, doc in enumerate(retrieved_docs_text)]
    )

    return context, retrieved_docs_text


def generate_answer(model, question: str, choices: List[str], retriever) -> str:
    contexts = retrieve_context(question, retriever)
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{choice}\n"
    prompt += f"""
    \nYou are a student that is solving the exam.
    Given contexts, please provide the letter (A, B, C, or D) of the correct answer based on the retrieved information.
    Context:
    {contexts}\n
    The response must follow the response format:
    - Return only one letter (A, B, C, or D)
    - No period or anything else at the end of the sentence\n
    Response format example 1:
    A
    Response format example 2:
    C
    Response format example 3:
    D
    """

    response = model.invoke(prompt)
    return response.strip()[-1]


def evaluate_performance(exam: List[Dict], results: List[str]) -> float:
    correct = sum(1 for q, r in zip(exam, results) if q["correct_answer"].startswith(r))
    return correct / len(exam)


def run_rag_exam(model_name: str, task_name: str, exam_file: str, retriever):
    exam = load_exam(exam_file)

    results = []
    for question in tqdm(exam, desc="Processing questions", unit="question"):
        try:
            answer = generate_answer(LLM, question["question"], question["choices"], retriever)
            results.append(answer)
        except:
            pass

    accuracy = evaluate_performance(exam, results)

    output = []
    for q, r in zip(exam, results):
        output.append(
            {
                "question": q["question"],
                "model_answer": r,
                "correct_answer": q["correct_answer"][0],
                "is_correct": r == q["correct_answer"][0],
            }
        )

    with open(f"Data/{task_name}/ExamResults/exam_results_{model_name}_{task_name}_{retriever}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exam completed. Accuracy: {accuracy:.2%}")
    print(f"Results saved to exam_results_{model_name}_{task_name}_{retriever}.json")


if __name__ == "__main__":
    model_name = "llamav2_rag"
    task_name = "StackExchange"
    exam_file = f"Data/{task_name}/ExamData/claude_gcp_2024100421/exam.json"
    corpus_file_name = "data_2024092613.json"
    retriever_type = "bm25"  # You can change this to "dpr" or "multiqa"

    # retriever_types = ["bm25", "dpr", "multiqa"]

    documents = load_corpus(f"Data/{task_name}/KnowledgeCorpus/main/{corpus_file_name}")

    text_splitter = CharacterTextSplitter(chunk_size=4500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    retriever = get_retriever(retriever_type)

    run_rag_exam(model_name, task_name, exam_file, retriever)
