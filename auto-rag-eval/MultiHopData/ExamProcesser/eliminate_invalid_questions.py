import json
from typing import List, Dict, Any
import logging

def validate_and_filter_questions(input_file: str, output_file: str) -> None:
    """
    Filters out JSON entries where the correct_answer is not one of 'A', 'B', 'C', or 'D'.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the filtered JSON file
        
    Raises:
        json.JSONDecodeError: If the input file is not valid JSON
        FileNotFoundError: If the input file doesn't exist
        IOError: If there are issues writing to the output file
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    valid_answers = {'A', 'B', 'C', 'D'}
    
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of dictionaries")
            
        initial_count = len(data)
        logger.info(f"Initial number of entries: {initial_count}")
        
        # Filter entries with valid correct answers
        filtered_data = [
            entry for entry in data 
            if isinstance(entry.get('correct_answer'), str) and
            entry.get('correct_answer').strip().upper() in valid_answers
        ]
        
        final_count = len(filtered_data)
        removed_count = initial_count - final_count
        logger.info(f"Removed {removed_count} entries with invalid answers")
        logger.info(f"Final number of entries: {final_count}")
        
        # Write the filtered data to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully wrote filtered data to {output_file}")
            
    except FileNotFoundError:
        logger.error(f"Input file {input_file} not found")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {str(e)}")
        raise
    except IOError as e:
        logger.error(f"Error writing to output file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    exam_file_names = ["exam_new_llama_3_2_3b", "exam_new_gemma2_9b", "exam_new_ministral_8b"]
    
    for task_domain in task_domains:
        for exam_file_name in exam_file_names:
            print(f"Processing {task_domain} - {exam_file_name}")
            input_file = f"auto-rag-eval/MultiHopData/{task_domain}/exams/{exam_file_name}_processed_v2.json"
            try:
                validate_and_filter_questions(
                    input_file=input_file,
                    output_file=input_file
                )
            except Exception as e:
                logging.error(f"Script failed: {str(e)}")
