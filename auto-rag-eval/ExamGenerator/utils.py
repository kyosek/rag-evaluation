import os
from collections import Counter
from typing import List

import json
import nltk
import numpy as np
from ExamGenerator.multi_choice_question import MultiChoiceQuestion
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("punkt_tab")


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
    all_data = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            # with open(file_path, 'r') as f:
            data = read_jsonl(file_path)

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
