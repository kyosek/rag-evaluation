import json
import random
from collections import Counter
from typing import List, Dict

def shuffle_exam_questions(input_file: str, output_file: str) -> None:
    """
    Reads exam questions from a JSON file, shuffles the choices to achieve even distribution
    of correct answers, and saves the result to a new JSON file.
    """
    # Read the JSON file
    with open(input_file, 'r') as f:
        questions = json.load(f)
    
    # Calculate target counts for even distribution
    total_questions = len(questions)
    target_per_option = total_questions // 4
    remaining = total_questions % 4
    
    # Create a list of positions to ensure even distribution
    positions = []
    for i in range(4):
        positions.extend([i] * target_per_option)
    # Distribute any remaining positions randomly
    if remaining:
        positions.extend(random.sample(range(4), remaining))
    
    # Shuffle the position list
    random.shuffle(positions)
    
    # Shuffle each question's choices
    shuffled_questions = []
    current_distribution = Counter()
    
    for question, position in zip(questions, positions):
        # Get the original choices and correct answer
        original_choices = question['choices']
        correct_answer = question['correct_answer']
        
        # Get the index of the correct answer (remove the trailing parenthesis)
        correct_index = ord(correct_answer[0]) - ord('A')
        correct_answer_text = original_choices[correct_index]
        
        # Create new shuffled choices with the correct answer in the assigned position
        new_choices = shuffle_other_choices(original_choices, correct_answer_text, position)
        
        # Update the question with new choices and correct answer
        new_question = question.copy()
        new_question['choices'] = new_choices
        new_question['correct_answer'] = f"{chr(65 + position)}"  # Convert 0-3 to A)-D)
        
        shuffled_questions.append(new_question)
        current_distribution[new_question['correct_answer']] += 1
    
    # Save the shuffled questions
    with open(output_file, 'w') as f:
        json.dump(shuffled_questions, f, indent=2)
    
    # Print distribution statistics
    print_distribution_stats(current_distribution)

def shuffle_other_choices(choices: List[str], correct_answer: str, correct_position: int) -> List[str]:
    """
    Shuffles the incorrect choices and places the correct answer in the specified position.
    """
    other_choices = [c for c in choices if c != correct_answer]
    random.shuffle(other_choices)
    
    # Insert correct answer at the specified position
    result = other_choices[:correct_position] + [correct_answer] + other_choices[correct_position:]
    return result[:4]  # Ensure we only return 4 choices

def print_distribution_stats(distribution: Counter) -> None:
    """
    Prints the distribution statistics of correct answers.
    """
    total = sum(distribution.values())
    print("\nCorrect Answer Distribution:")
    for answer, count in sorted(distribution.items()):
        percentage = (count / total) * 100
        print(f"{answer}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    input_file = "auto-rag-eval/MultiHopData/hotpotqa/exams/llama_3_2_3b_single_hop_exam_cleaned_1000_42.json"
    output_file = "auto-rag-eval/MultiHopData/hotpotqa/exams/llama_3_2_3b_single_hop_exam_cleaned_shuffled_1000_42.json"
    
    shuffle_exam_questions(input_file, output_file)
