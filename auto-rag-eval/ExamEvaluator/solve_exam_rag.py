from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, DPRRetriever, MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
from typing import Dict, List

from LLMServer.llama.llama_instant import LlamaModel

LLM = LlamaModel()
EMBEDDINGS = HuggingFaceEmbeddings()


def extract_text(data: Dict) -> List[str]:
    return [data.get("text", "")]


def load_corpus(data_path: str):
    loader = JSONLoader(
        file_path=data_path,
        jq_schema='.[]',
        content_key="text",
        text_content=False
    )
    return loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Initialize vector store
vectorstore = Chroma.from_documents(texts, EMBEDDINGS)

# Initialize retrievers
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 5  # Number of documents to retrieve

dpr_retriever = DPRRetriever(
    retriever=vectorstore.as_retriever(),
    embeddings=embeddings
)

multiqa_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

# Function to switch between retrievers
def get_retriever(retriever_type):
    if retriever_type == "bm25":
        return bm25_retriever
    elif retriever_type == "dpr":
        return dpr_retriever
    elif retriever_type == "multiqa":
        return multiqa_retriever
    else:
        raise ValueError("Invalid retriever type")

# Function to perform RAG
def rag_query(query, retriever_type):
    retriever = get_retriever(retriever_type)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    return result["result"], result["source_documents"]


if __name__ == "__main__":

    retriever_types = ["bm25", "dpr", "multiqa"]

    documents = load_corpus(f"Data/{task_name}/KnowledgeCorpus/main/{file_name}")

    for retriever_type in retriever_types:
        print(f"\nUsing {retriever_type.upper()} retriever:")
        answer, sources = rag_query(query, retriever_type)
        print(f"Answer: {answer}")
        print("Sources:")
        for i, doc in enumerate(sources):
            print(f"  {i+1}. {doc.page_content[:100]}...")