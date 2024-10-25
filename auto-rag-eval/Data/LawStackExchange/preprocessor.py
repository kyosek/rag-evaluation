import re
from typing import List, Dict
from datetime import datetime
import time
from functools import reduce
from os.path import abspath, dirname
from tqdm import tqdm
from datasets import load_dataset, Dataset
from markdownify import markdownify as md

ROOTPATH = dirname(dirname(abspath(__file__)))


class LawStackExchangeDataLoader:
    def __init__(self, n_samples: int, max_char_length: int):
        self.n_samples = n_samples
        self.max_char_length = max_char_length

    def get_best_answer(self, answers: List[Dict[str, str]]):
        """Return the best answer, that is, the one with the highest positive score"""
        if not answers:
            return None
        positive_answers = [a for a in answers if int(a['score']) > 0]
        if not positive_answers:
            return None
        return max(positive_answers, key=lambda x: int(x['score']))['body']

    def lang_callback(self, el):
        lang = el["class"][0] if el.has_attr("class") else None
        return lang.split("-")[-1] if lang else None

    def html2md(self, text: str) -> str:
        text = md(text, code_language_callback=self.lang_callback)
        text = re.sub(r"\n\s*\n", "\n\n", text).strip()
        return text.encode("utf-8", "replace").decode()

    def process_qna(self, row):
        return {
            "source": row["link"],
            "docs_id": row["question_id"],
            "title": row["question_title"],
            "section": "N/A",
            "date": "N/A",
            "start_character": "N/A",
            "end_character": "N/A",
            "text": self.html2md(
                f"### User: {row['question_body']} -\n\n### Top Answer: {self.get_best_answer(row['answers'])}"
            ),
        }

    def load_save_dataset(self) -> None:
        # Load the Law-StackExchange dataset
        dataset = load_dataset("ymoslem/Law-StackExchange", split="train")

        funcs = [
            # Select subset of data and preprocess to keep only top answer
            lambda data: data.shuffle(seed=42).select(range(min(len(data), self.n_samples))).map(
                self.process_qna,
                remove_columns=['question_id', 'link', 'question_title', 'question_body', 'answers', 'license', 'tags', 'score']
            ),
            # Remove too lengthy answers
            lambda data: data.filter(lambda x: len(x['text']) <= self.max_char_length)
        ]

        filtered_dataset = reduce(lambda res, f: f(res), funcs, dataset)

        # Save the processed dataset
        filtered_dataset.to_json(
            f"{ROOTPATH}/LawStackExchange/KnowledgeCorpus/main/data_{datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H')}.json",
            lines=False
        )


if __name__ == "__main__":
    loader = LawStackExchangeDataLoader(n_samples=10000, max_char_length=10000)
    loader.load_save_dataset()
