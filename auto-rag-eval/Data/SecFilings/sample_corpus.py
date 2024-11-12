import json
import random


def sample_json(filename, sample_size, output_filename, seed=42):
    """Randomly samples a specified number of dictionaries from a JSON file and saves them to a new JSON file.

    Args:
      filename: The path to the input JSON file.
      sample_size: The desired number of samples.
      output_filename: The path to the output JSON file.
      seed: The seed value for the random number generator.
    """

    random.seed(seed)

    with open(filename, "r") as f:
        data = json.load(f)

    # Randomly sample the desired number of dictionaries
    random_sample = random.sample(data, sample_size)

    with open(output_filename, "w") as f:
        json.dump(random_sample, f, indent=4)


if __name__ == "__main__":
    # Replace 'your_json_file.json' and 'sampled_data.json' with your desired file paths
    filename = "auto-rag-eval/Data/SecFilings/KnowledgeCorpus/main/data_2024102022_original.json"
    sample_size = 10000
    output_filename = "auto-rag-eval/Data/SecFilings/KnowledgeCorpus/main/data_2024102022.json"
    seed = 42  # You can change the seed value to get different random samples

    sample_json(filename, sample_size, output_filename, seed)
