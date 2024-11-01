import os
from collections import Counter
from typing import List, Dict, Union, Optional

import chardet
import json
import nltk
import numpy as np
from ExamGenerator.multi_choice_question import MultiChoiceQuestion
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("punkt_tab")




def robust_json_load(filepath: str) -> List[Dict]:
    """
    Robustly load JSON files with multiple fallback mechanisms
    
    Args:
        filepath (str): Path to the JSON file
    
    Returns:
        List[Dict]: Parsed JSON data
    """
    def detect_encoding(filepath):
        """Detect file encoding"""
        with open(filepath, 'rb') as file:
            result = chardet.detect(file.read())
        return result['encoding']
    
    def read_file_with_encoding(filepath, encoding):
        """Read file with specified encoding"""
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read().strip()
        return content
    
    def parse_json_with_fallbacks(content):
        """Try multiple JSON parsing strategies"""
        parse_attempts = [
            # Standard JSON parsing
            lambda: json.loads(content),
            
            # Remove potential BOM (Byte Order Mark)
            lambda: json.loads(content.lstrip('\ufeff')),
            
            # Try to fix common JSON formatting issues
            lambda: json.loads(content.replace('\n', '').replace('\r', '')),
            
            # Try to parse lines individually
            lambda: [json.loads(line.strip()) for line in content.split('\n') if line.strip()]
        ]
        
        for attempt in parse_attempts:
            try:
                result = attempt()
                # Ensure result is a list
                return result if isinstance(result, list) else [result]
            except (json.JSONDecodeError, TypeError):
                continue
        
        # If all attempts fail
        raise ValueError(f"Unable to parse JSON from file: {filepath}")
    
    try:
        # First, try standard JSON loading
        with open(filepath, 'r') as f:
            return json.load(f)
    
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Initial JSON loading failed: {e}")
        
        # Try detecting encoding
        try:
            detected_encoding = detect_encoding(filepath)
            print(f"Detected encoding: {detected_encoding}")
            
            # Read with detected encoding
            content = read_file_with_encoding(filepath, detected_encoding)
            
            # Parse with multiple fallback strategies
            return parse_json_with_fallbacks(content)
        
        except Exception as encoding_error:
            print(f"Encoding-based loading failed: {encoding_error}")
            
            # Last resort: read raw content and attempt parsing
            try:
                with open(filepath, 'rb') as f:
                    raw_content = f.read().decode('utf-8', errors='ignore')
                return parse_json_with_fallbacks(raw_content)
            
            except Exception as final_error:
                print(f"Final JSON loading attempt failed: {final_error}")
                raise ValueError(f"Absolutely unable to load JSON from {filepath}")


def read_jsonl(file_path: str):
    flattened_data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                json_object = json.loads(line)
                # Extend the flattened_data list with the contents of json_object
                # This assumes json_object is always a list of dictionaries
                flattened_data.extend(json_object)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}")
                print(f"Error message: {str(e)}")
    return flattened_data



def flatten_data(data_folder: str) -> list:
    """
    Flatten JSON data with robust handling of different field types
    """
    # all_data = []
    # for filename in os.listdir(data_folder):
    #     if filename.endswith('.json'):
    #         with open(os.path.join(data_folder, filename), 'r') as f:
    #             json_data = json.load(f)
                
    #             for key, item in json_data.items():
    #                 # Normalize section to a string
    #                 section = item['documentation'].get('section', 'N/A')
    #                 if isinstance(section, list):
    #                     section = ', '.join(section)
                    
    #                 # Create a flattened entry
    #                 entry = {
    #                     'source': item['documentation'].get('source', 'N/A'),
    #                     'docs_id': item['documentation'].get('docs_id', 'N/A'),
    #                     'title': item['documentation'].get('title', 'N/A'),
    #                     'section': section,
    #                     'start_character': item['documentation'].get('start_character', 'N/A'),
    #                     'end_character': item['documentation'].get('end_character', 'N/A'),
    #                     'text': item['documentation'].get('text', ''),
    #                     'date': item['documentation'].get('date', 'N/A'),
    #                     'answer': item.get('answer', '')
    #                 }
    #                 all_data.append(entry)
    
    # return all_data
    all_data = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            robust_json_load(file_path)
            # with open(file_path, 'r') as f:
            #     data = json.load(f)
            # data = read_jsonl(file_path)

            for item in data:
                all_data.append(
                    {
                        "title": item["title"],
                        "source": item["source"],
                        "docs_id": item["docs_id"],
                        "section": item["section"],
                        "start_character": item["start_character"],
                        "end_character": item["end_character"],
                        "date": item["date"],
                        "text": item["text"],
                    }
                )

    return all_data


def get_n_sentences(text):
    return len([sent for sent in nltk.sent_tokenize(text) if len(sent) > 5])


def get_single_file_in_folder(folder_path):
    # List all entries in the given folder
    entries = os.listdir(folder_path)

    # Filter out only the files (excluding directories and other types)
    files = [
        os.path.join(folder_path, f)
        for f in entries
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # Check the number of files
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        raise ValueError(f"No files found in the directory {folder_path}")
    else:
        raise ValueError(
            f"More than one file found in the directory {folder_path}. Files are: {', '.join(files)}"
        )


class SimilarityChecker:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def preprocess_text(self, text: str) -> int:
        text = text.lower()
        word_count = Counter(text.split())
        return word_count

    def jaccard_similarity(self, counter1: Counter, counter2: Counter) -> float:
        intersection = sum((counter1 & counter2).values())
        union = sum((counter1 | counter2).values())
        return intersection / union

    def calculate_max_similarity(self, sentence: List[str], reference_doc: str) -> float:
        similarities = [
            self.jaccard_similarity(
                self.preprocess_text(main_sentence), self.preprocess_text(sentence)
            )
            for main_sentence in sent_tokenize(reference_doc)
        ]
        return max(similarities)

    def get_ngrams(self, text: str, n: int) -> List[str]:
        words = text.split()
        return [" ".join(words[i : i + n]) for i in range(len(words) - (n - 1))]

    def calculate_max_ngram_similarity(
        self, sentence: List[str], reference_doc: str, n: int
    ) -> float:
        main_ngrams = self.get_ngrams(reference_doc, n)
        similarities = [
            self.jaccard_similarity(
                self.preprocess_text(main_ngram), self.preprocess_text(sentence)
            )
            for main_ngram in main_ngrams
        ]
        return max(similarities, default=0)

    def calculate_embedding_similarity(self, sentence: List[str], mcq: MultiChoiceQuestion):
        main_text_embedding = self.model.encode([mcq.documentation])
        sentence_embeddings = self.model.encode([sentence])
        return cosine_similarity([main_text_embedding[0]], [sentence_embeddings[0]])[0][0]

    def compute_similarity(self, mcq: MultiChoiceQuestion) -> List[str]:
        mean_ngram = int(np.mean([len(answer.split()) for answer in mcq.choices]))
        return [
            (
                f"{self.calculate_max_similarity(answer, mcq.documentation):.02f}"
                f"{self.calculate_max_ngram_similarity(answer, mcq.documentation, mean_ngram):.02f}"
                f"{self.calculate_embedding_similarity(answer, mcq):.02f}"
            )
            for answer in mcq.choices
        ]
