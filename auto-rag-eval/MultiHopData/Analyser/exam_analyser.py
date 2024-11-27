import json
import numpy as np
import random
import os
from typing import List, Dict
from collections import Counter
from dataclasses import dataclass
import re
import argparse


@dataclass
class MCQuestion:
    question: str
    choices: List[str]
    correct_answer: str
    documentation: List[str]

    def correct_candidate_is_longest(self) -> bool:
        """Check if the correct answer is the longest among all choices."""
        correct_idx = ord(self.correct_answer[0]) - ord("A")
        lengths = [len(choice) for choice in self.choices]
        return lengths[correct_idx] == max(lengths)

    def display(self):
        """Display the question details."""
        print(f"\nQuestion: {self.question}")
        for choice in self.choices:
            print(choice)
        print(f"Correct Answer: {self.correct_answer}")
        print("Documentation:")
        for doc in self.documentation:
            print(doc)


class ExamAnalyser:
    def __init__(self, exam_data: List[Dict], task_domain: str):
        self.task_domain = task_domain
        self.question_date = self.get_current_date()

        # Initialize analysis metrics
        self.question_list: List[MCQuestion] = []
        self.failed_question_list = []
        self.n_question = len(exam_data)

        # Parse questions
        self.parse_questions(exam_data)

        # Error counters
        self.question_parsing_fail = 0
        self.choices_parsing_fail = 0
        self.correct_answer_parsing_fail = 0
        self.other_parsing_fail = 0

    @staticmethod
    def get_current_date():
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d")

    @staticmethod
    def get_n_sentences(text: str) -> int:
        """Count the number of sentences in a text."""
        return len(re.split(r"[.!?]+", text))

    def parse_questions(self, exam_data: List[Dict]) -> None:
        """Parse exam questions into MCQuestion objects."""
        for question_data in exam_data:
            try:
                # Extract correct answer letter from the full answer string
                correct_answer = question_data["correct_answer"].split(")")[0]

                question = MCQuestion(
                    question=question_data["question"],
                    choices=question_data["choices"],
                    correct_answer=correct_answer,
                    documentation=question_data["documentation"],
                )
                self.question_list.append(question)
            except Exception as e:
                self.failed_question_list.append({"question_data": question_data, "error": str(e)})
                self.other_parsing_fail += 1

    def compute_exam_analytics(
        self, save_failed_question: bool = True, display_n_samples: int = 1
    ) -> Dict:
        """Compute and display exam analytics."""
        if self.n_question == 0:
            raise ValueError("Empty exam, please check if exam data was loaded properly.")

        if save_failed_question and self.failed_question_list:
            output_path = f"failed_questions_{self.task_domain}_{self.question_date}.json"
            with open(output_path, "w") as outfile:
                json.dump(self.failed_question_list, outfile)

        def convert_perc(x):
            return 100 * x / len(self.question_list)

        # Collect analytics
        analytics = {}

        # Basic stats
        analytics["exam_id"] = f"{self.task_domain}:{self.question_date}"
        analytics["total_questions"] = self.n_question
        analytics["processed_questions"] = len(self.question_list)
        analytics["processing_success_rate"] = 100 * len(self.question_list) / self.n_question

        # Answer analysis
        answer_analysis = Counter([question.correct_answer[0] for question in self.question_list])
        analytics["answer_distribution"] = dict(answer_analysis)
        analytics["best_fixed_answer_baseline"] = convert_perc(max(answer_analysis.values()))
        analytics["longest_answer_baseline"] = convert_perc(
            sum([mcq.correct_candidate_is_longest() for mcq in self.question_list])
        )

        # Question analysis
        question_keyword = ["Which", "What", "How", "When", "Why", "Where"]
        question_types = {}
        for k in question_keyword:
            count = sum([k.lower() in mcq.question.lower() for mcq in self.question_list])
            question_types[k] = convert_perc(count)

        other_key = sum(
            [
                not (any([k.lower() in mcq.question.lower() for k in question_keyword]))
                for mcq in self.question_list
            ]
        )
        question_types["Other"] = convert_perc(other_key)
        analytics["question_types"] = question_types

        # Length analysis
        analytics["avg_question_length"] = {
            "mean": float(np.mean([len(mcq.question) for mcq in self.question_list])),
            "std": float(np.std([len(mcq.question) for mcq in self.question_list])),
        }
        analytics["avg_answer_length"] = {
            "mean": float(np.mean([len("".join(mcq.choices)) / 4 for mcq in self.question_list])),
            "std": float(np.std([len("".join(mcq.choices)) / 4 for mcq in self.question_list])),
        }
        analytics["avg_documentation_length"] = {
            "mean": float(np.mean([len("".join(mcq.documentation)) for mcq in self.question_list])),
            "std": float(np.std([len("".join(mcq.documentation)) for mcq in self.question_list])),
        }

        # Log analytics
        self._log_analytics(analytics)

        # Display sample questions if requested
        if display_n_samples > 0:
            print("\nSample Questions:")
            for elem in random.sample(
                self.question_list, min(display_n_samples, len(self.question_list))
            ):
                elem.display()

        return analytics

    def _log_analytics(self, analytics: Dict) -> None:
        """Log the analytics results."""
        print(f"\n########################################################\n")
        print(f"ExamID: {analytics['exam_id']}")
        print(
            f"\nProcessing Analysis:\n"
            f"Total of {analytics['processed_questions']}/{analytics['total_questions']} "
            f"questions processed ({analytics['processing_success_rate']:.2f}%)"
        )

        print(
            f"\nAccuracy Analysis:\n"
            f"Best Fixed Answer Baseline: {analytics['best_fixed_answer_baseline']:.2f}%\n"
            f"Longest Answer Baseline: {analytics['longest_answer_baseline']:.2f}%"
        )

        print("\nQuestion Type Analysis:")
        for qtype, percentage in analytics["question_types"].items():
            print(f"{qtype:7} -- {percentage:.2f}%")

        print(
            f"\nLength Analysis:\n"
            f"Avg. question length: {analytics['avg_question_length']['mean']:.2f} "
            f"(std: {analytics['avg_question_length']['std']:.2f})\n"
            f"Avg. answer length: {analytics['avg_answer_length']['mean']:.2f} "
            f"(std: {analytics['avg_answer_length']['std']:.2f})\n"
            f"Avg. documentation length: {analytics['avg_documentation_length']['mean']:.2f} "
            f"(std: {analytics['avg_documentation_length']['std']:.2f})"
        )


def main(exam_path: str, task_domain: str):
    parser = argparse.ArgumentParser(description="Analyse multiple choice exam data")
    parser.add_argument(
        "--display-samples", type=int, default=3, help="Number of sample questions to display"
    )
    parser.add_argument("--output-dir", default=None, help="Directory to save analytics output")

    args = parser.parse_args()

    # Load exam data
    with open(exam_path, "r") as f:
        exam_data = json.load(f)

    # Create analyzer and compute analytics
    analyzer = ExamAnalyser(exam_data, task_domain)
    analytics = analyzer.compute_exam_analytics(
        save_failed_question=True, display_n_samples=args.display_samples
    )

    # Save analytics
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"exam_analytics_{task_domain}.json")
    else:
        output_path = os.path.splitext(exam_path)[0] + "_analytics.json"

    with open(output_path, "w") as f:
        json.dump(analytics, f, indent=2)

    print(f"\nAnalytics saved to: {output_path}")


if __name__ == "__main__":
    task_domain = "gov_report"
    
    exam_path = f"auto-rag-eval/MultiHopData/gov_report/exams/exam_new_llama_3_2_3b_processed.json"

    main(exam_path, task_domain)
