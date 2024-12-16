import json
import argparse
from pathlib import Path

def clean_json_results(input_path: str, output_path: str) -> tuple[int, int]:
    """
    Clean JSON results by removing invalid questions that start with 'A)'.
    
    Args:
        input_path (str): Path to the input JSON file
        output_path (str): Path where the cleaned JSON will be saved
        
    Returns:
        tuple[int, int]: Number of total entries and number of valid entries
    """
    # Read the input JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Count total entries
    total_entries = len(results)
    
    # Filter out invalid questions
    valid_results = [
        entry for entry in results 
        if not entry['question'].strip().startswith('A)')
    ]
    
    # Count valid entries
    valid_entries = len(valid_results)
    
    # Save the cleaned results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(valid_results, f, indent=2, ensure_ascii=False)
        
    return total_entries, valid_entries

def main(input_path: str):
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Clean JSON results by removing invalid questions')
    parser.add_argument('input_path', help='Path to the input JSON file')
    parser.add_argument(
        '--output_path', 
        help='Path for the output JSON file (default: input_path with _cleaned suffix)',
        default=None
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output_path is None:
        input_path = Path(args.input_path)
        output_path = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")
    else:
        output_path = args.output_path
    
    # Process the file
    total_entries, valid_entries = clean_json_results(args.input_path, output_path)
    
    # Print summary
    print(f"Processing complete!")
    print(f"Total entries: {total_entries}")
    print(f"Valid entries: {valid_entries}")
    print(f"Removed entries: {total_entries - valid_entries}")
    print(f"Cleaned results saved to: {output_path}")

if __name__ == "__main__":
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    exam_files = [
        "exam_new_llama_3_2_3b_processed_v3.json",
        "exam_new_gemma2_9b_processed_v2.json",
        "exam_new_ministral_8b_processed_v3.json",
        ]
    
    for task_domain in task_domains:
        for exam_file in exam_files:
            main()
