import os
import json
import logging
import re
import time
from datetime import datetime
from functools import reduce
from os.path import abspath, dirname
from typing import List

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
ROOTPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(f"{ROOTPATH}/Feverous/feverous_categories.json", "r") as f:
    CATEGORIES = {elem["tag"]: elem["name"] for elem in json.load(f) if elem["tag"] != "N/A"}


class ArxivData:

    def __init__(self, n_samples: int, max_char_length: int, min_char_length: int):

        self.n_samples = n_samples
        self.max_char_length = max_char_length
        self.min_char_length = min_char_length
        self.cat_list = ["Business", "Entertainment", "Health", "Science", "Sports", "Technology"]

    # TODO: Add date and start/end character
    def process_qna(self, row):

        return {
            "source": row["authors"],
            "docs_id": row["id"],
            "title": row["title"],
            "section": self.get_full_cat_list(row["categories"]),
            "start_character": "N/A",
            "end_character": "N/A",
            "date": "N/A",
            "text": f"{row['title']}. {self.preprocess(row['abstract'])}",
        }

    def preprocess(self, text: str) -> str:

        text = text.replace("\n", " ").replace("\r", "").strip()
        text = re.sub(r" +", " ", text)
        return text

    def load_save_data(self) -> None:

        dataset = load_dataset(
            path="hotpotqa/hotpot_qa",
            name="fullwiki",
            split="train",
        )

        # Remove too lengthy or shorty answers to avoid repeting operation
        sub_df = dataset.filter(
            lambda x: self.min_char_length <= len(x["context"]["sentences"]) <= self.max_char_length
        )
        logger.info(
            (
                f"Reducing dataset size from {len(dataset)} to {len(sub_df)} by keeping context with"
                f" character length between {self.min_char_length} and {self.max_char_length}."
            )
        )

        # Join over all categories at the end and shuffle again
        def funcs(category):
            return [
                # Filter only for a given category
                lambda data: data.filter(
                    lambda x: any([tag == category for tag in x["categories"]])
                ),
                # Select Subset of data and preprocess to keep only top answer
                lambda data: data.shuffle(seed=42)
                .select(range(min(len(data), self.n_samples)))
                .map(
                    self.process_qna,
                    remove_columns=[
                        "id",
                        "submitter",
                        "authors",
                        "title",
                        "comments",
                        "journal-ref",
                        "doi",
                        "abstract",
                        "report-no",
                        "categories",
                        "versions",
                    ],
                ),
            ]

        data_cat_list = []

        for category in tqdm(self.cat_list):
            data_cat_list.append(reduce(lambda res, f: f(res), funcs(category), sub_df))
        concat_dataset = concatenate_datasets(data_cat_list).shuffle(seed=42)

        concat_dataset.to_json(
            f"{ROOTPATH}/Feverous/KnowledgeCorpus/main/data_{datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H')}.json",
            lines=False,
        )


if __name__ == "__main__":

    feverous_data = ArxivData(n_samples=1000, max_char_length=1500, min_char_length=1000)

    feverous_data.load_save_data()
