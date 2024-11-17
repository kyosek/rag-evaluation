import json
from collections import defaultdict

def calculate_accuracy_by_hops(data):
    """
    Calculate accuracy grouped by number of hops from a list of question data.
    
    Args:
        data (list): List of dictionaries containing question data
        
    Returns:
        dict: Dictionary with number_of_hops as keys and accuracy as values
    """
    # Initialize counters for total questions and correct answers by hops
    totals_by_hops = defaultdict(int)
    correct_by_hops = defaultdict(int)
    
    # Count totals and correct answers for each number of hops
    for item in data:
        hops = item['number_of_hops']
        totals_by_hops[hops] += 1
        if item['is_correct']:
            correct_by_hops[hops] += 1
    
    # Calculate accuracy for each number of hops
    accuracy_by_hops = {}
    for hops in totals_by_hops:
        accuracy = correct_by_hops[hops] / totals_by_hops[hops]
        accuracy_by_hops[hops] = accuracy
    
    # Add summary statistics
    total_questions = sum(totals_by_hops.values())
    total_correct = sum(correct_by_hops.values())
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    
    return {
        'accuracy_by_hops': accuracy_by_hops,
        'overall_accuracy': overall_accuracy,
        'total_questions': total_questions,
        'questions_by_hops': dict(totals_by_hops)
    }

# Example usage:
if __name__ == "__main__":
    # Read JSON file
    with open('auto-rag-eval/MultiHopData/gov_report/MISTRAL_7B_closed_exam_results.json', 'r') as f:
        data = json.load(f)
    
    # Calculate accuracies
    results = calculate_accuracy_by_hops(data)
    
    # Print results
    print("\nAccuracy by number of hops:")
    for hops, accuracy in sorted(results['accuracy_by_hops'].items()):
        print(f"{hops} hops: {accuracy:.2%} ({results['questions_by_hops'][hops]} questions)")
    
    print(f"\nOverall accuracy: {results['overall_accuracy']:.2%}")
    print(f"Total questions: {results['total_questions']}")
