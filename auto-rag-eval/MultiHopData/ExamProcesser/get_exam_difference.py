import json
from typing import List, Dict, Any
import argparse
from pathlib import Path

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON file and return its content.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        List[Dict[str, Any]]: List of exam questions
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_filtered_questions(original_exam: List[Dict[str, Any]], 
                            filtered_exam: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract questions that were filtered out by comparing original and filtered exams.
    
    Args:
        original_exam (List[Dict[str, Any]]): Original exam questions
        filtered_exam (List[Dict[str, Any]]): Filtered exam questions
        
    Returns:
        List[Dict[str, Any]]: List of filtered out questions
    """
    # Create sets of questions for efficient comparison
    original_questions = {q['question'] for q in original_exam}
    filtered_questions = {q['question'] for q in filtered_exam}
    
    # Find questions that were filtered out
    filtered_out_questions = original_questions - filtered_questions
    
    # Get the complete question entries for filtered out questions
    filtered_entries = [
        q for q in original_exam 
        if q['question'] in filtered_out_questions
    ]
    
    return filtered_entries

def save_json(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save data to a JSON file with proper formatting.
    
    Args:
        data (List[Dict[str, Any]]): Data to save
        output_path (str): Path where to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main(original_exam: str, filtered_exam: str, output_path: str) -> None:
    try:
        # Load both exam files
        original_exam = load_json(original_exam)
        filtered_exam = load_json(filtered_exam)
        
        # Extract filtered questions
        filtered_questions = extract_filtered_questions(original_exam, filtered_exam)
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        save_json(filtered_questions, output_path)
        print(f"Successfully extracted {len(filtered_questions)} filtered questions")
        print(f"Results saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")


if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    exams = [
        "llama_3_2_3b",
        "ministral_8b",
        "gemma2_9b",
        ]
    
    for task in task_domains:
        for exam in exams:
            original_exam = f"auto-rag-eval/MultiHopData/{task}/exams/exam_new_{exam}.json"
            filtered_exam = f"auto-rag-eval/MultiHopData/{task}/exams/exam_new_{exam}_processed_v2.json"
            output_path = f"auto-rag-eval/MultiHopData/{task}/exams/exam_new_{exam}_v1_v2_diff.json"
            
            main(original_exam, filtered_exam, output_path)
