import json
import os


def update_json_files(directory_path):
    """
    Update JSON files in the specified directory by adding "date": "N/A" to each documentation section.

    Args:
        directory_path (str): Path to the directory containing JSON files
    """
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            try:
                # Read the JSON file
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                # Update each numbered key's documentation section
                for key in data:
                    if "documentation" in data[key]:
                        data[key]["documentation"]["date"] = "N/A"

                # Write the updated data back to the file
                with open(file_path, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=4)

                print(f"Successfully updated {filename}")

            except json.JSONDecodeError:
                print(f"Error: {filename} is not a valid JSON file")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


def update_knowledge_corpus_json_file(directory_path):
    """
    Update JSON files in the specified directory by adding "date": "N/A" to each object.

    Args:
        directory_path (str): Path to the directory containing JSON files
    """
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            try:
                # Read the JSON file
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                # Check if the data is a list
                if isinstance(data, list):
                    # Update each object in the array
                    for item in data:
                        if isinstance(item, dict):
                            item["date"] = "N/A"

                # Write the updated data back to the file
                with open(file_path, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=4)

                print(f"Successfully updated {filename}")

            except json.JSONDecodeError:
                print(f"Error: {filename} is not a valid JSON file")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


# Usage example
if __name__ == "__main__":
    # Replace with your directory path
    # directory_path = "Data/LawStackExchange/RawExamData"
    # update_json_files(directory_path)
    directory_path = "Data/LawStackExchange/KnowledgeCorpus/main"
    update_knowledge_corpus_json_file(directory_path)
