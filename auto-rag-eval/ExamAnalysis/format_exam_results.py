import json
import os
from typing import List, Dict
import argparse


def convert_exam_result(input_data: List[Dict]) -> List[Dict]:
    """
    Convert exam results from the original format to the IRT model format.

    Args:
        input_data: List of dictionaries containing exam results in original format

    Returns:
        List of dictionaries in the required IRT model format
    """
    converted_data = []

    for question in input_data:
        converted_question = {
            "acc": 1.0 if question["is_correct"] else 0.0,
            "doc": {
                "question": question["question"]
            },
            # Preserving original data as additional fields
            "metadata": {
                "model_answer": question["model_answer"],
                "correct_answer": question["correct_answer"]
            }
        }
        converted_data.append(converted_question)

    return converted_data


def main(input_dir: str, output_dir: str) -> None:
    """
    Process all JSON files in a directory structure and convert them to the required format.

    Args:
        input_dir: Path to input directory containing exam results
        output_dir: Path to output directory for converted files
    """
    # Walk through directory structure
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                # Create corresponding output directory structure
                rel_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                os.makedirs(output_path, exist_ok=True)

                # Process file
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_path, file.replace('.json', '.jsonl'))

                try:
                    # Read input file
                    with open(input_file, 'r') as f:
                        input_data = json.load(f)

                    # Convert data
                    converted_data = convert_exam_result(input_data)

                    # Write output file
                    with open(output_file, 'w') as f:
                        for item in converted_data:
                            json.dump(item, f)
                            f.write('\n')

                    print(f"Successfully converted {input_file} to {output_file}")

                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert exam results to IRT model format')
    parser.add_argument('--input', required=True, help='Input directory containing exam results')
    parser.add_argument('--output', required=True, help='Output directory for converted files')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    main(args.input, args.output)
