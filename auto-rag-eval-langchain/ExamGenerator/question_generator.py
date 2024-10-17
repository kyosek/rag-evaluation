import argparse
import concurrent.futures
import json
import logging
import random
import time
import os
from datetime import datetime
from os.path import abspath, dirname
from typing import List

# from ExamGenerator.utils import read_jsonl
from LLMServer.llama.llama_instant import LlamaModel
from LLMServer.gcp.claude_instant import Claude_GCP
from LLMServer.llm_exam_generator import LLMExamGenerator, LlamaExamGenerator, ClaudeExamGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOTPATH = dirname(dirname(abspath(__file__)))


def read_jsonl(file_path: str):
    flattened_data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                json_object = json.loads(line)
                flattened_data.extend(json_object)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}")
                print(f"Error message: {str(e)}")
    return flattened_data


class BatchExamGenerator:

    def __init__(self, data_path: str, task_domain: str, model_list: List[str], batch_size: int):

        self.data_path = data_path
        self.batch_size = batch_size
        self.model_list = model_list
        self.task_domain = task_domain

        self.model_map = {
            "llama": LlamaExamGenerator(
                step_size=1, task_domain=self.task_domain, llm_model=LlamaModel()
            ),
            "claude_gcp": ClaudeExamGenerator(
                step_size=1, task_domain=self.task_domain, llm_model=Claude_GCP()
            ),
        }
        assert not (any([model not in self.model_map.keys() for model in self.model_list]))

    def batch_generate_exam(self, data_folder: str) -> None:

        data = read_jsonl(data_folder)

        random.seed(42)
        random.shuffle(data)

        logger.info(
            (
                f"Processing a total of {len(data)} documentation pieces for {self.task_domain}"
                f" using models {self.model_list}, with batch size of {self.batch_size} "
                f"({1+len(data)//self.batch_size} batches)"
            )
        )

        # Split the data into batches
        batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]

        start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H")

        # for batch_index, batch in enumerate(batches):
        for batch_index, batch in enumerate(batches):

            logger.info(f"Running batch {batch_index}.")
            if len(self.model_list) > 1:
                # Multiprocessing not compatible with Bedrock Usage
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = {
                        model: executor.submit(self.model_map[model].generate_exam, batch)
                        for model in self.model_list
                    }
                    generated_questions = {model: future.result() for model, future in futures.items()}
            else:
                generated_questions = {
                    model: self.model_map[model].generate_exam(batch) for model in self.model_list
                }

            # Write the dictionary to a JSON file
            for model in generated_questions.keys():
                filename = f"{self.task_domain}_QCM_{model}_{start_time}_batch{batch_index}.json"
                full_path = os.path.join(
                    ROOTPATH, "Data", self.task_domain, "RawExamData", filename
                )

                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                with open(
                    f"{ROOTPATH}/Data/{self.task_domain}/RawExamData/{filename}", "w"
                ) as write_file:
                    json.dump(generated_questions[model], write_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Creates Raw Exam from Documentation Corpus")

    parser.add_argument(
        "--task-domain",
        help="Task Domain, among DevOps, StackExchange, MyOwnTask...",
    )
    parser.add_argument(
        "--file-name",
        help="File name to the KnowledgeCorpus",
        default="data_2024092613.json",
    )

    main_args, _ = parser.parse_known_args()

    raw_exam_generator = BatchExamGenerator(
        data_path=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main/",
        batch_size=64,
        task_domain=main_args.task_domain,
        model_list=["llama"],
    )

    raw_exam_generator.batch_generate_exam(
        data_folder=f"{ROOTPATH}/Data/{main_args.task_domain}/KnowledgeCorpus/main/{main_args.file_name}"
    )
