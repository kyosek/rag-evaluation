import json
from typing import List, Dict, Any


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


def create_question_hop_mapping(reference_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Create a mapping of questions to their number of hops using num_chunks_used.
    
    Args:
        reference_data: List of dictionaries containing the reference data with metadata
        
    Returns:
        Dictionary mapping questions to their number of hops (num_chunks_used)
    """
    mapping = {}
    for item in reference_data:
        try:
            question = item['question']
            num_chunks = item['metadata']['num_chunks_used']
            mapping[question] = num_chunks
        except KeyError as e:
            print(f"Warning: Missing required field in reference data: {e}")
    return mapping


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


def main(reference_file: str, target_file: str, output_file: str) -> None:
    """
    Main function to process the JSON files.
    
    Args:
        reference_file: Path to the reference JSON file with metadata containing num_chunks_used
        target_file: Path to the JSON file needing number_of_hops
        output_file: Path where to save the updated JSON file
    """
    try:
        # Load both JSON files
        print(f"Loading files...")
        reference_data = load_json_file(reference_file)
        target_data = load_json_file(target_file)
        
        # Create mapping of questions to number of hops
        print("Creating question-hop mapping from num_chunks_used...")
        question_hop_mapping = create_question_hop_mapping(reference_data)
        
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
        pass


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    model_names = ['llama_3_1_8b', "ministral-8b", "gemma2-27b",]
    exam_model_names = ["llama_3_2_3b", "gemma2_9b", 'ministral_8b']
    
    for task_domain in task_domains:
        for model_name in model_names:
            for exam_model_name in exam_model_names:
                # correct_file = f"auto-rag-eval/MultiHopData/{task_domain}/exams/exam_new_{exam_model_name}_processed_v2.json"
                correct_file = f"auto-rag-eval/MultiHopData/{task_domain}/exams/exam_new_{exam_model_name}_processed_v2_unfiltered.json"
                target_file = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{model_name}_open_exam_new_{exam_model_name}_processed_v2.json.json"
                # output_file = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{model_name}_open_exam_new_{exam_model_name}_processed_v2.json.json"
                
                main(correct_file, target_file, target_file)
