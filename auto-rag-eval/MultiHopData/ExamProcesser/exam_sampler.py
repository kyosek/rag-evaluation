import json
import random


def sample_json_entries(input_file, output_file, sample_size=1000, random_seed=42):
    """
    Randomly sample entries from a JSON file with a fixed seed.

    Parameters:
    input_file (str): Path to the input JSON file
    output_file (str): Path where the output JSON file will be saved
    sample_size (int): Number of entries to sample (default: 1000)
    random_seed (int): Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Read the JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure the sample size isn't larger than the dataset
    if sample_size > len(data):
        print(f"Warning: Sample size ({sample_size}) is larger than the dataset size ({len(data)})")
        sample_size = len(data)

    # Randomly sample entries
    sampled_data = random.sample(data, sample_size)

    # Save the sampled entries to a new JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully sampled {sample_size} entries from {len(data)} total entries")
    print(f"Output saved to: {output_file}")


# Example usage
if __name__ == "__main__":
    sample_size = 1000
    seed = 42
    # input_file = "Data/gov_report/ExamData/claude_gcp_2024103117/exam.json"
    # output_file = f"Data/gov_report/ExamData/claude_gcp_2024103117/exam_{sample_size}_{seed}.json"
    input_file = "auto-rag-eval/MultiHopData/gov_report/exams/exam_new_gemma2_9b_cleaned.json"
    output_file = f"auto-rag-eval/MultiHopData/gov_report/exams/gemma2_9b_exam_cleaned_{sample_size}_{seed}.json"

    sample_json_entries(input_file, output_file, sample_size)
