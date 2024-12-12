import json
from typing import List, Dict, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing the JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
        raise

def match_questions(updated_exam: List[Dict], original_results: List[Dict]) -> List[Dict]:
    """Match questions between updated exam and original results.
    
    Args:
        updated_exam: List of questions from the updated exam
        original_results: List of results from the original exam
        
    Returns:
        List of matched results
    """
    # Create a map of questions to their results for faster lookup
    original_map = {result['question']: result for result in original_results}
    matched_results = []
    
    for question in updated_exam:
        if question['question'] in original_map:
            matched_results.append(original_map[question['question']])
            
    logger.info(f"Matched {len(matched_results)} questions out of {len(updated_exam)} questions in updated exam")
    return matched_results

def calculate_accuracy(results: List[Dict]) -> float:
    """Calculate accuracy from exam results.
    
    Args:
        results: List of exam results
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if not results:
        return 0.0
        
    correct_count = sum(1 for result in results if result['is_correct'])
    accuracy = correct_count / len(results)
    return accuracy

def save_results(results: List[Dict], output_path: str):
    """Save results to a JSON file.
    
    Args:
        results: List of results to save
        output_path: Path where to save the results
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except IOError as e:
        logger.error(f"Error saving results: {e}")
        raise

def process_exam_results(
    updated_exam_path: str,
    original_results_path: str,
    output_path: str
) -> float:
    """Process exam results and calculate new accuracy.
    
    Args:
        updated_exam_path: Path to the updated exam JSON file
        original_results_path: Path to the original results JSON file
        output_path: Path where to save the new results
        
    Returns:
        Calculated accuracy
    """
    # Load data
    updated_exam = load_json_file(updated_exam_path)
    original_results = load_json_file(original_results_path)
    
    # Match questions and calculate accuracy
    matched_results = match_questions(updated_exam, original_results)
    accuracy = calculate_accuracy(matched_results)
    
    # Save results
    save_results(matched_results, output_path)
    
    logger.info(f"Accuracy: {accuracy:.2%}")
    logger.info(f"Number of questions in matched results: {len(matched_results)}")
    
    return accuracy

if __name__ == "__main__":
    # task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    task_domains = ["hotpotqa"]
    exam_file_names = ["exam_new_llama_3_2_3b", "exam_new_ministral_8b"]
    exam_types = ["closed", "open", "Dense", "Sparse", "Hybrid", "Rerank"]
    model_names = [
        'llama_3_1_8b',
        "ministral-8b",
        "gemma2-9b",
    ]
    
    for task_domain in task_domains:
        for exam_file_name in exam_file_names:
            for model_name in model_names:
                for exam_type in exam_types:
                    print(f"Processing {task_domain} - {exam_file_name} - {exam_type} - {model_name}")
                    
                    updated_exam_path = f"auto-rag-eval/MultiHopData/{task_domain}/exams/{exam_file_name}_processed_v3.json"
                    if exam_types in ["closed", "open"]:
                        original_results_path = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{model_name}_{exam_type}_{exam_file_name}_processed_v2.json.json"
                        output_path = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{model_name}_{exam_type}_{exam_file_name}_processed_v3.json.json"
                    else:
                        original_results_path = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{model_name}_{exam_type}_{exam_file_name}_processed_v2.json_5_results.json"
                        output_path = f"auto-rag-eval/MultiHopData/{task_domain}/exam_results/{model_name}_{exam_type}_{exam_file_name}_processed_v3.json_5_results.json"
                    
                    try:
                        accuracy = process_exam_results(
                            updated_exam_path,
                            original_results_path,
                            output_path
                        )
                    except Exception as e:
                        logger.error(f"Error processing exam results: {e}")
