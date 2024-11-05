import os
import json
import uuid
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


class DocumentChunker:
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200):
        """Initialize the DocumentChunker with configurable parameters.
        
        Args:
            chunk_size: The size of each chunk in characters
            chunk_overlap: The overlap between chunks in characters
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single text file and return its chunks in the desired format."""
        # Generate a doc_id from the filename
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load and process the document
        loader = TextLoader(file_path)
        document = loader.load()[0]
        
        # Get the full content
        full_content = document.page_content
        
        # Generate chunks
        chunk_texts = self.text_splitter.split_text(full_content)
        
        # Create chunks with metadata
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            chunk = {
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "original_index": idx,
                "content": chunk_text
            }
            chunks.append(chunk)
        
        # Create the document entry
        doc_entry = {
            "doc_id": doc_id,
            "original_uuid": uuid.uuid4().hex[:4],
            "content": full_content,
            "chunks": chunks
        }
        
        return doc_entry
    
    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Process all text files in a directory and return a list of processed documents."""
        processed_docs = []
        
        # Process each .txt file in the directory
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(dir_path, filename)
                doc_entry = self.process_file(file_path)
                processed_docs.append(doc_entry)
        
        return processed_docs
    
    def save_to_json(self, processed_docs: List[Dict[str, Any]], output_path: str):
        """Save the processed documents to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_docs, f, indent=2, ensure_ascii=False)

def main(task_name: str):
    # Example usage
    chunker = DocumentChunker(chunk_size=4000, chunk_overlap=200)
    
    # Process a directory of text files
    docs_dir = f"auto-rag-eval/MultiHopData/{task_name}"
    output_path = "docs_chunk.json"
    
    # Process all documents
    processed_docs = chunker.process_directory(docs_dir)
    
    # Save to JSON file
    chunker.save_to_json(processed_docs, output_path)
    
    print(f"Processed {len(processed_docs)} documents")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    task_name = "wiki"
    
    main(task_name)
