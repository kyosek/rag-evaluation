import json
import random
import copy
from typing import List, Dict, Any
from collections import Counter

def balance_exam_answers(exam_data: List[Dict[Any, Any]], target_distribution: Dict[str, int] = None) -> List[Dict[Any, Any]]:
    """
    Balance the distribution of correct answers in the exam while maintaining 
    the integrity of each question's choices.
    
    Args:
        exam_data (List[Dict]): List of exam questions
        target_distribution (Dict, optional): Desired distribution of answers
    
    Returns:
        List[Dict]: Balanced exam questions
    """
    # Make a deep copy to avoid modifying the original data
    balanced_exam = copy.deepcopy(exam_data)
    
    # Calculate initial distribution
    initial_distribution = get_answer_distribution(balanced_exam)
    print("Initial Answer Distribution:")
    print(json.dumps(initial_distribution, indent=2))
    
    # If no target distribution is provided, create an even distribution
    if target_distribution is None:
        total_questions = len(balanced_exam)
        equal_per_option = total_questions // 4
        remainder = total_questions % 4
        
        target_distribution = {
            'A': equal_per_option + (1 if remainder > 0 else 0),
            'B': equal_per_option + (1 if remainder > 1 else 0),
            'C': equal_per_option + (1 if remainder > 2 else 0),
            'D': equal_per_option
        }
    
    # Collect questions by their current correct answer
    answer_buckets = {answer: [] for answer in 'ABCD'}
    for question in balanced_exam:
        answer_buckets[question['correct_answer']].append(question)
    
    # Balanced distribution algorithm
    balanced_questions = []
    
    for target_answer, target_count in target_distribution.items():
        # Get questions with this current answer
        current_bucket = answer_buckets[target_answer]
        
        if len(current_bucket) > target_count:
            # If we have too many, randomly remove excess
            random.shuffle(current_bucket)
            balanced_questions.extend(current_bucket[:target_count])
        elif len(current_bucket) < target_count:
            # If we need more, borrow from other buckets
            balanced_questions.extend(current_bucket)
            needed = target_count - len(current_bucket)
            
            # Collect potential questions to redistribute
            redistribution_pool = []
            for other_answer in 'ABCD':
                if other_answer != target_answer:
                    redistribution_pool.extend(answer_buckets[other_answer])
            
            # Shuffle and select additional questions
            random.shuffle(redistribution_pool)
            for question in redistribution_pool:
                if needed > 0:
                    # Modify the question's correct answer
                    original_correct_index = question['choices'].index(
                        next(choice for choice in question['choices'] if choice.startswith(question['correct_answer'] + ')'))
                    )
                    
                    # Change the correct answer to the target
                    question['correct_answer'] = target_answer
                    balanced_questions.append(question)
                    needed -= 1
                
                if needed == 0:
                    break
        else:
            # Exact match, just add all questions
            balanced_questions.extend(current_bucket)
    
    # Verify final distribution
    final_distribution = get_answer_distribution(balanced_questions)
    print("\nFinal Answer Distribution:")
    print(json.dumps(final_distribution, indent=2))
    
    return balanced_questions

def get_answer_distribution(exam_data: List[Dict[Any, Any]]) -> Dict[str, int]:
    """
    Calculate the distribution of correct answers in the exam.
    
    Args:
        exam_data (List[Dict]): List of exam questions
    
    Returns:
        Dict[str, int]: Distribution of correct answers
    """
    distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    for question in exam_data:
        correct_answer = question['correct_answer']
        distribution[correct_answer] += 1
    
    return distribution

def balance_exam_file(input_file: str, output_file: str, seed: int = None, 
                       target_distribution: Dict[str, int] = None):
    """
    Read exam questions from a JSON file, balance them, and write to a new file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file
        seed (int, optional): Random seed for reproducibility
        target_distribution (Dict, optional): Desired answer distribution
    """
    # Set random seed if provided for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Read the exam data
    with open(input_file, 'r', encoding='utf-8') as f:
        exam_data = json.load(f)
    
    # Balance the exam questions
    balanced_exam = balance_exam_answers(exam_data, target_distribution)
    
    # Write the balanced exam to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_exam, f, indent=2, ensure_ascii=False)
    
    print(f"\nBalanced exam saved to {output_file}")


if __name__ == "__main__":
    input_file = "auto-rag-eval/MultiHopData/hotpotqa/exams/llama_3_2_3b_exam_cleaned_1000_42.json"
    output_file = "auto-rag-eval/MultiHopData/hotpotqa/exams/llama_3_2_3b_exam_cleaned_shuffled_1000_42.json"
    
    balance_exam_file(input_file, output_file, seed=42)
    