import json
from typing import List, Dict, Any
import argparse
from pathlib import Path


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing the JSON data
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_question_hop_mapping(correct_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Create a mapping of questions to their number of hops.
    
    Args:
        correct_data: List of dictionaries containing the correct data with number_of_hops
        
    Returns:
        Dictionary mapping questions to their number of hops
    """
    return {item['question']: item['number_of_hops'] for item in correct_data}


def update_json_with_hops(target_data: List[Dict[str, Any]], 
                         question_hop_mapping: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Update the target JSON data with number_of_hops field.
    
    Args:
        target_data: List of dictionaries to update
        question_hop_mapping: Mapping of questions to their number of hops
        
    Returns:
        Updated list of dictionaries
    """
    updated_data = []
    unmatched_questions = []
    
    for item in target_data:
        question = item['question']
        if question in question_hop_mapping:
            item['number_of_hops'] = question_hop_mapping[question]
            updated_data.append(item)
        else:
            unmatched_questions.append(question)
            updated_data.append(item)  # Keep the original item without number_of_hops
    
    if unmatched_questions:
        print(f"\nWarning: {len(unmatched_questions)} questions couldn't be matched:")
        for q in unmatched_questions[:5]:  # Show first 5 unmatched questions
            print(f"- {q[:100]}...")  # Show first 100 characters
        if len(unmatched_questions) > 5:
            print(f"... and {len(unmatched_questions) - 5} more")
    
    return updated_data


def save_json_file(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save the updated data to a JSON file.
    
    Args:
        data: List of dictionaries to save
        output_path: Path where to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main(correct_file: str, target_file: str, output_file: str) -> None:
    """
    Main function to process the JSON files.
    
    Args:
        correct_file: Path to the JSON file with correct number_of_hops
        target_file: Path to the JSON file needing number_of_hops
        output_file: Path where to save the updated JSON file
    """
    try:
        # Load both JSON files
        print(f"Loading files...")
        correct_data = load_json_file(correct_file)
        target_data = load_json_file(target_file)
        
        # Create mapping of questions to number of hops
        print("Creating question-hop mapping...")
        question_hop_mapping = create_question_hop_mapping(correct_data)
        
        # Update target data with number_of_hops
        print("Updating target data with number_of_hops...")
        updated_data = update_json_with_hops(target_data, question_hop_mapping)
        
        # Save the updated data
        print(f"Saving updated data to {output_file}...")
        save_json_file(updated_data, output_file)
        
        print("\nProcess completed successfully!")
        print(f"Total questions processed: {len(target_data)}")
        print(f"Questions matched and updated: {sum(1 for item in updated_data if 'number_of_hops' in item)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    model_names = ['llama_3_1_8b', "ministral-8b", "gemma2-27b",]
    exam_model_names = ["llama_3_2_3b", "gemma2_9b", 'ministral_8b']
    
    for task_domain in task_domains:
        for model_name in model_names:
            for exam_model_name in exam_model_names:
                correct_file = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{model_name}_closed_exam_new_{exam_model_name}_processed_v2_unfiltered.json.json"
                target_file = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{model_name}_open_exam_new_{exam_model_name}_processed_v2.json.json"
                
                main(correct_file, target_file, target_file)
