import os
import json
from typing import List, Dict
# from LLMServer.llama.llama_instant import LlamaModel
from LLMServer.llama_gcp.llama_gcp_instant import LlamaGcpModel
from tqdm import tqdm


# Load the JSON file
def load_exam(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Generate an answer for a given question
def generate_answer(model, question: str, choices: List[str], document) -> str:
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{choice}\n"
    prompt += f"\nDocument:\n {document}"
    prompt += """
    \nYou are a student that is solving the exam.
    Please solve the question by using the given document and 
    provide the letter (A, B, C, or D) only of the correct answer.\n
    The response must follow the response format:
    - Return only one letter (A, B, C, or D)
    - No period or anything else at the end of the sentence\n
    Response format example 1:
    A
    Response format example 2:
    C
    Response format example 3:
    D
    """

    response = model.invoke(prompt)
    return response.strip()[-1]


# Evaluate the model's performance
def evaluate_performance(exam: List[Dict], results: List[str]) -> float:
    correct = sum(1 for q, r in zip(exam, results) if q["correct_answer"].startswith(r))
    return correct / len(exam)


# Main function to run the exam
def run_open_book_exam(model_device: str, model_path: str, model_name: str, task_name: str, exam_file: str):
    exam = load_exam(exam_file)
    if model_device == "GCP":
        print("Using transformer")
        model = LlamaGcpModel(
            model_size="70B",
            use_gpu=True,
            load_in_4bit=True
            )
    else:
        print("Using Llama-cpp")
        model = LlamaModel(model_path=model_path)

    results = []
    for question in tqdm(exam, desc="Processing questions", unit="question"):
        answer = generate_answer(model, question["question"], question["choices"], question["documentation"])
        results.append(answer)

    accuracy = evaluate_performance(exam, results)

    output = []
    for q, r in zip(exam, results):
        output.append(
            {
                "question": q["question"],
                "model_answer": r,
                "correct_answer": q["correct_answer"][0],
                "is_correct": r == q["correct_answer"][0],
            }
        )

    with open(f"Data/{task_name}/ExamResults/open_exam_results_{model_name}_{task_name}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exam completed. Accuracy: {accuracy:.2%}")
    print(f"Results saved to exam_results_{model_name}.json")


if __name__ == "__main__":
    model_device = "GCP"
    model_path = "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF"
    # model_path = "Meta-Llama-3.1-70B-Instruct-Q4_K_S.gguf"
    model_name = "llamav2"
    # model_name = "llama3-B70"
    task_name = "LawStackExchange"
    exam_file = f"Data/{task_name}/ExamData/claude_gcp_2024102123/exam_1000_42.json"

    # Create the full directory path
    directory = f"Data/{task_name}/ExamResults"
    os.makedirs(directory, exist_ok=True)

    run_open_book_exam(model_device, model_path, model_name, task_name, exam_file)
