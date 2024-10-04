import json


def find_json_error(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    for i in range(len(data)):
        try:
            json.loads(data[:i + 1])
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                print(f"Error at position {i}:")
                print(data[max(0, i - 50):i + 50])  # Print context around the error
                break


if __name__ == "__main__":
    find_json_error('Data/StackExchange/KnowledgeCorpus/main/data_2024092613.json')
