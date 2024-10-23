import argparse
from typing import List
import os
import numpy as np
from ExamAnalysis.item_response_models import ExamSetting
from ExamAnalysis.iterative_item_response_models import IterativeHierarchicalItemResponseModel


def load_exam_settings(base_path: str) -> List[ExamSetting]:
    """
    Load exam settings for different configurations of LLM, retrieval, and ICL.

    Args:
        base_path: Base path where exam data is stored

    Returns:
        List of ExamSetting objects
    """
    llm_models = ["llamav2"]
    retrieval_methods = ["BM25", "DPR", "SIAMESE", "MultiQA", "DPR:MultiQA:BM25"]
    icl_shots = [0]

    exam_settings = []
    for llm in llm_models:
        for retrieval in retrieval_methods:
            for icl in icl_shots:
                data_path = os.path.join(base_path, f"rag_exam_results_{llm}_{retrieval}.jsonl")
                if os.path.exists(data_path):
                    exam_settings.append(
                        ExamSetting(
                            name=f"{llm}_{retrieval}_icl{icl}",
                            llm=llm,
                            retrieval=retrieval,
                            icl=icl,
                            path_pattern=data_path
                        )
                    )
    return exam_settings


def main(task_domain: str):
    # Configuration
    base_data_path = f"Data/{task_domain}/FormattedExamResults/"
    irt_model_type = 1  # Using 3PL model (1, 2 or 3)
    n_iterations = 3  # Number of iteration steps
    drop_ratio = 0.1  # Ratio of questions to drop in each iteration

    # Load exam settings
    exam_settings = load_exam_settings(base_data_path)

    if not exam_settings:
        print("No exam settings found. Please check the data path.")
        return

    # Initialize and fit the model
    model = IterativeHierarchicalItemResponseModel(
        students=exam_settings,
        irt_model_type=irt_model_type
    )

    # Fit the model and get estimators for each iteration
    estimator_dict = model.fit(n_steps=n_iterations, drop_ratio=drop_ratio)

    # Plot the iterative informativeness
    model.plot_iterative_informativeness(
        estimator_dict=estimator_dict,
        exam_model="Your Exam Name",
        save_path="exam_information_curve.png",
        font_size=14
    )

    # Print statistics for each iteration
    for step, estimator in estimator_dict.items():
        print(f"\nStats for Iteration {step}:")
        stats = model.compute_stats(estimator)

        print(f"Number of questions: {len(estimator['discrimination'])}")
        print(
            f"Mean exam accuracy: {stats['Mean Exam accuracy']['mean']:.2f}% (±{stats['Mean Exam accuracy']['std']:.2f}%)")
        print("\nEstimator Statistics:")
        for param, values in stats['Estimators'].items():
            print(f"{param}: {values['mean']:.3f} (±{values['std']:.3f})")

        # Save the iterated exam for each configuration
        if step > 0:  # Skip saving for initial step
            # Calculate indices of questions to keep
            sorted_indices = np.argsort(estimator['discrimination'])
            kept_indices = sorted_indices[int(len(sorted_indices) * drop_ratio):]
            model.save_iterated_exam(step, kept_indices)

    print("\nExam iteration completed. Check the 'Data/IteratedExams' directory for the results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates Raw Exam from Documentation Corpus")

    parser.add_argument(
        "--task-domain",
        help="Task Domain, among DevOps, StackExchange, MyOwnTask...",
    )

    main_args, _ = parser.parse_known_args()

    main(main_args.task_domain)
