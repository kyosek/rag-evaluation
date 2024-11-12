import json
from copy import deepcopy


def needs_cleaning(question):
    """
    Check if a question needs cleaning based on its choices.

    Args:
        question (dict): Question dictionary

    Returns:
        bool: True if question needs cleaning, False otherwise
    """
    if "choices" not in question:
        return True

    choices = question["choices"]
    valid_options = ["A) ", "B) ", "C) ", "D) "]

    # Check if exactly 4 choices
    if len(choices) != 4:
        return True

    # Check if all choices start with correct prefixes
    for i, choice in enumerate(choices):
        if not choice.startswith(valid_options[i]):
            return True

    # Check if correct_answer needs cleaning
    if "correct_answer" in question:
        answer = question["correct_answer"].strip().upper()
        if len(answer) > 1 or answer not in ["A", "B", "C", "D"]:
            return True

    return False


def clean_question_choices(data):
    """
    Clean question data while maintaining the original format.

    Args:
        data (list): List of question dictionaries

    Returns:
        list: List of cleaned and unmodified questions in original order
    """
    cleaned_data = []
    questions_cleaned = 0

    for question in data:
        if needs_cleaning(question):
            # Create a clean copy of the question
            cleaned_question = deepcopy(question)

            # Ensure choices list exists
            if "choices" not in cleaned_question:
                cleaned_question["choices"] = []

            # Get current choices
            choices = cleaned_question["choices"]

            # Clean and standardize the choices
            cleaned_choices = []
            valid_options = ["A) ", "B) ", "C) ", "D) "]

            # Keep only the first 4 choices and ensure they match the format
            for i, choice in enumerate(choices[:4]):
                if i < len(valid_options):
                    # If the choice doesn't start with the correct prefix, add it
                    if not choice.startswith(valid_options[i]):
                        choice = valid_options[i] + choice.lstrip("ABCD) ")
                    cleaned_choices.append(choice)

            # If we have fewer than 4 choices, add empty ones
            while len(cleaned_choices) < 4:
                cleaned_choices.append(valid_options[len(cleaned_choices)])

            # Update the choices in the cleaned question
            cleaned_question["choices"] = cleaned_choices

            # Clean the correct answer format if needed
            if "correct_answer" in cleaned_question:
                answer = cleaned_question["correct_answer"].strip().upper()
                # Extract just the letter if it includes more
                answer = answer[0] if answer else ""
                cleaned_question["correct_answer"] = answer

            cleaned_data.append(cleaned_question)
            questions_cleaned += 1
        else:
            # Keep the original question unchanged
            cleaned_data.append(question)

    return cleaned_data, questions_cleaned


def process_json_file(input_path, output_path):
    """
    Read JSON file, clean the data, and save it back to a new JSON file.

    Args:
        input_path (str): Path to the input JSON file
        output_path (str): Path where the cleaned JSON file will be saved
    """
    try:
        # Read the input JSON file
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Clean the data and get statistics
        cleaned_data, questions_cleaned = clean_question_choices(data)

        # Save the cleaned data to a new JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully processed {input_path}")
        print(f"- Total questions: {len(data)}")
        print(f"- Questions cleaned: {questions_cleaned}")
        print(f"- Questions unmodified: {len(data) - questions_cleaned}")
        print(f"Results saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Could not find the input file {input_path}")
    except json.JSONDecodeError:
        print(f"Error: The file {input_path} is not valid JSON")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    input_file = "MultiHopData/SecFilings/exam_new.json"
    output_file = "MultiHopData/SecFilings/exam_cleaned.json"
    process_json_file(input_file, output_file)
