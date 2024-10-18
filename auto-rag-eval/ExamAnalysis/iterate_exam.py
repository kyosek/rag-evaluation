import json
import os
from os.path import abspath, dirname
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ExamAnalysis.item_response_models import BaseItemResponseModel, ExamSetting
from ExamAnalysis.iterative_item_response_models import IterativeHierarchicalItemResponseModel


def get_exam_setting_from_path(exam_data_path: str, icl: int) -> ExamSetting:
    """
    Creates an ExamSetting object from the given exam data path and ICL value.
    You'll need to adapt the logic to extract LLM and retrieval information from the path.
    """
    # Example logic (adapt as needed):
    parts = exam_data_path.split('/')
    llm = f"{parts[-3]}:{parts[-2]}"  # Assuming LLM is in the second-to-last part
    retrieval = parts[-4]  # Assuming retrieval is in the third-to-last part
    name = f"{retrieval}@{icl} [{llm}]"  # Create a descriptive name

    return ExamSetting(
        path_pattern=exam_data_path,
        llm=llm,
        retrieval=retrieval,
        icl=icl,
        name=name
    )


def print_nested_dict(d: dict, indent: int = 0):
    """Recursively prints nested dictionaries with increasing indentation."""
    for key, value in d.items():
        print('   ' * indent + str(key))
        if isinstance(value, dict):
            print_nested_dict(value, indent + 1)
        else:
            print('   ' * (indent + 1) + (f"{value:.02f}" if type(value) != str else value))


def run_iterative_irt(exam_data_path: str, irt_model_type: int, n_steps: int, drop_ratio: float):
    """Runs the iterative IRT analysis and saves results."""
    # Create ExamSetting objects for different ICL values (adjust range as needed)
    exam_settings = [get_exam_setting_from_path(exam_data_path, i) for i in range(3)]

    # Extract task and LLM model from the path (adapt as needed)
    parts = exam_data_path.split('/')
    task = parts[-5]  # Assuming task is in the fifth-to-last part
    llm_model = parts[-3]  # Assuming LLM model is in the third-to-last part

    print(f'Starting Analysis for task {task}, llm: {llm_model} and irt {irt_model_type}')
    expe_name = f"{llm_model}_recursive_irt_{irt_model_type}"

    iterative_item_response_analyzer = IterativeHierarchicalItemResponseModel(
        students=exam_settings,
        irt_model_type=irt_model_type
    )
    estimator_dict = iterative_item_response_analyzer.fit(n_steps=n_steps, drop_ratio=drop_ratio)
    all_stats = {
        step_k: iterative_item_response_analyzer.compute_stats(estimator_dict[step_k])
        for step_k in estimator_dict.keys()
    }

    task_path = f"{dirname(dirname(abspath(__file__)))}/Data/{task}/EvalResults/IterativeIRT"
    iterative_item_response_analyzer.plot_iterative_informativeness(
        estimator_dict=estimator_dict,
        exam_model=f'{task}:{llm_model.capitalize()}',
        save_path=f"{task_path}/18_{task}_fig_{expe_name}_step{n_steps}.png"
    )

    with open(f"{task_path}/recursive_irt_step{n_steps}.json", "w") as outfile:
        outfile.write(json.dumps(all_stats))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run iterative IRT analysis.')
    parser.add_argument('exam_data_path', type=str, help='Path to the exam data file.')
    parser.add_argument('--irt_model_type', type=int, default=3, help='IRT model type (1, 2, or 3)')
    parser.add_argument('--n_steps', type=int, default=4, help='Number of iterative steps')
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='Drop ratio for low-discrimination questions')

    args = parser.parse_args()

    run_iterative_irt(args.exam_data_path, args.irt_model_type, args.n_steps, args.drop_ratio)
