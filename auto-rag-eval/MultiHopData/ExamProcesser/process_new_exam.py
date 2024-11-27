import json
import random
import pandas as pd
from collections import Counter

def process_json_data(input_file, sample_size, random_seed=42):
    """
    Process JSON data with filtering, cleaning, and balanced sampling.
    
    Args:
        input_file (str): Path to input JSON file
        sample_size (int): Desired size of the final sample
        random_seed (int): Random seed for reproducibility
        
    Returns:
        list: Processed and sampled JSON entries
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Read JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 1. Filter entries where meets_hop_requirement is True
    filtered_data = [
        entry for entry in data 
        if entry.get('metadata', {}).get('meets_hop_requirement', False)
    ]
    
    # 2. Clean correct_answer by removing ")"
    for entry in filtered_data:
        if 'correct_answer' in entry:
            try:
                entry['correct_answer'] = entry['correct_answer'].replace(')', '')
            except:
                pass
    
    # 3. Balanced sampling based on num_chunks_used
    # Get distribution of num_chunks_used
    chunk_counts = Counter(
        entry['metadata']['num_chunks_used'] 
        for entry in filtered_data
    )
    
    # Calculate target number per chunk_size category
    num_categories = len(chunk_counts)
    target_per_category = sample_size // num_categories
    remainder = sample_size % num_categories
    
    # Group entries by num_chunks_used
    grouped_entries = {}
    for entry in filtered_data:
        num_chunks = entry['metadata']['num_chunks_used']
        if num_chunks not in grouped_entries:
            grouped_entries[num_chunks] = []
        grouped_entries[num_chunks].append(entry)
    
    # Sample from each group
    sampled_data = []
    for chunks, entries in grouped_entries.items():
        # Add one extra to some categories if we have remainder
        current_target = target_per_category
        if remainder > 0:
            current_target += 1
            remainder -= 1
            
        # Sample min between target and available entries
        sample_count = min(current_target, len(entries))
        sampled_data.extend(random.sample(entries, sample_count))
    
    # If we still haven't reached sample_size, sample randomly from remaining entries
    if len(sampled_data) < sample_size:
        remaining_needed = sample_size - len(sampled_data)
        remaining_entries = [
            entry for entry in filtered_data 
            if entry not in sampled_data
        ]
        if remaining_entries:
            additional_samples = random.sample(
                remaining_entries, 
                min(remaining_needed, len(remaining_entries))
            )
            sampled_data.extend(additional_samples)
    
    return sampled_data


def print_distribution(data):
    """Print the distribution of num_chunks_used in the data."""
    chunks_dist = Counter(
        entry['metadata']['num_chunks_used'] 
        for entry in data
    )
    print("\nDistribution of num_chunks_used:")
    for chunks, count in sorted(chunks_dist.items()):
        print(f"{chunks} chunks: {count} entries")


if __name__ == "__main__":
    task_domain = "gov_report"
    file_name = "exam_new_gemma2_9b"
    
    input_file = f"auto-rag-eval/MultiHopData/{task_domain}/exams/{file_name}.json"  # Replace with your file path
    sample_size = 600  # Replace with desired sample size
    
    # Process the data
    processed_data = process_json_data(input_file, sample_size)
    
    # Print results
    print(f"Total entries after processing: {len(processed_data)}")
    print_distribution(processed_data)
    
    # Optionally save processed data
    with open(f"auto-rag-eval/MultiHopData/{task_domain}/exams/{file_name}_processed_v2_unfiltered.json", 'w') as f:
        json.dump(processed_data, f, indent=2)
