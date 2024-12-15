import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging


class ExamCleaner:
    """A class to clean exam questions and generate summary statistics.
    
    This class processes exam JSON files by removing invalid questions and 
    generating summary statistics about the cleaning process.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize the ExamCleaner with custom logging configuration.
        
        Args:
            log_level: The logging level to use (default: logging.INFO)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def clean_exam(self, exam_path: str) -> Tuple[Dict, Dict]:
        """Clean the exam by removing invalid questions and generate summary.
        
        Args:
            exam_path: Path to the JSON file containing exam questions
            
        Returns:
            Tuple containing:
                - Dictionary with cleaned exam data
                - Dictionary with summary statistics
                
        Raises:
            FileNotFoundError: If the exam file doesn't exist
            json.JSONDecodeError: If the exam file contains invalid JSON
        """
        try:
            self.logger.info(f"Processing exam file: {exam_path}")
            
            # Read the exam file
            with open(exam_path, 'r', encoding='utf-8') as f:
                exam_data = json.load(f)
            
            # Extract questions (handle both list and dict formats)
            questions = (exam_data['questions'] 
                       if isinstance(exam_data, dict) 
                       else exam_data)
            
            original_count = len(questions)
            self.logger.info(f"Original question count: {original_count}")
            
            # Filter out invalid questions
            valid_questions = [
                q for q in questions 
                if not str(q.get('question', '')).strip().startswith('A) ')
            ]
            
            cleaned_count = len(valid_questions)
            eliminated_count = original_count - cleaned_count
            
            self.logger.info(f"Eliminated {eliminated_count} questions")
            self.logger.info(f"Remaining questions: {cleaned_count}")
            
            # Prepare cleaned exam data
            cleaned_exam = (
                {**exam_data, 'questions': valid_questions}
                if isinstance(exam_data, dict)
                else valid_questions
            )
            
            # Generate summary
            summary = {
                'original_count': original_count,
                'eliminated_count': eliminated_count,
                'final_count': cleaned_count,
                'exam_path': exam_path
            }
            
            return cleaned_exam, summary
            
        except FileNotFoundError:
            self.logger.error(f"Exam file not found: {exam_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in exam file: {exam_path}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing exam: {str(e)}")
            raise

    def save_results(
        self, 
        cleaned_exam: Dict,
        summary: Dict,
        input_dir: str,
        output_dir: str
    ) -> None:
        """Save the cleaned exam and summary to files.
        
        Args:
            cleaned_exam: The cleaned exam data
            summary: Summary statistics of the cleaning process
            output_dir: Directory to save the output files
            
        Raises:
            OSError: If there's an error creating the output directory or saving files
        """
        try:
            # Save files
            with open(input_dir, 'w', encoding='utf-8') as f:
                json.dump(cleaned_exam, f, indent=2, ensure_ascii=False)
            
            with open(output_dir, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
                
            self.logger.info(f"Saved cleaned exam to: {input_dir}")
            self.logger.info(f"Saved summary to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

def validate_and_filter_questions(input_file: str, output_file: str, summary_output_path: str) -> None:
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
    
    cleaner = ExamCleaner()
    
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
        
        cleaned_exam, summary = cleaner.clean_exam(input_file)
        cleaner.save_results(cleaned_exam, summary, input_file, summary_output_path)
            
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
    # exam_file_names = ["exam_new_llama_3_2_3b", "exam_new_gemma2_9b", "exam_new_ministral_8b"]
    exam_file_names = ["exam_new_gemma2_9b"]
    
    for task_domain in task_domains:
        for exam_file_name in exam_file_names:
            print(f"Processing {task_domain} - {exam_file_name}")
            input_file = f"auto-rag-eval/MultiHopData/{task_domain}/exams/{exam_file_name}_processed_v5.json"
            summary_output_path = f"auto-rag-eval/MultiHopData/{task_domain}/exam_stats/{exam_file_name}_processed_v5_stats.json"
            try:
                validate_and_filter_questions(
                    input_file=input_file,
                    output_file=input_file,
                    summary_output_path=summary_output_path
                )
            except Exception as e:
                logging.error(f"Script failed: {str(e)}")
