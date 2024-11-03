import os
import json
import torch

from typing import List, Dict
from LLMServer.llama.llama_instant import LlamaModel
from LLMServer.llama_gcp.llama_gcp_instant import LlamaGcpModel
from LLMServer.gcp.claude_instant import Claude_GCP
from tqdm import tqdm

model_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "device_map": "auto"
}


# Load the JSON file
def load_exam(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Generate an answer for a given question
def generate_answer(model: LlamaModel, question: str, choices: List[str]) -> str:
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}) {choice}\n"
    prompt += """
    \nYou are a student that is solving the exam.
    Please provide the letter (A, B, C, or D) only of the correct answer.
    The response must follow the response format:
    Response format example:
    A
    """

    response = model.invoke(prompt)
    return response.strip()[-1]


def generate_answer_llama(model, question: str, choices: List[str]) -> str:
    # Format choices with letters for clear instruction
    formatted_choices = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
    
    # Construct a more structured prompt with system and user roles
    prompt = f"""[INST] <<SYS>>
    You are an AI assistant taking a multiple choice exam. Your task is to:
    1. Read the question and provided document carefully
    2. Analyze the choices
    3. Select the most appropriate answer
    4. Respond with ONLY the letter (A, B, C, or D) of the correct answer
    <</SYS>>

    Question: {question}

    Choices:
    {formatted_choices}

    Instructions:
    - You must respond with exactly one letter: A, B, C, or D
    - Do not include any explanation, period, or additional text
    - Just the letter of the correct answer

    Examples of valid responses:
    A
    B
    C
    D

    Your answer (one letter only): [/INST]"""

    # Get model response
    response = model.invoke(prompt)
    
    # Extract just the letter from the response
    # Look for first occurrence of A, B, C, or D
    valid_answers = {'A', 'B', 'C', 'D'}
    for char in response:
        if char in valid_answers:
            return char
            
    # If no valid letter found, return the last character as fallback
    return response.strip()[-1]


# Evaluate the model's performance
def evaluate_performance(exam: List[Dict], results: List[str]) -> float:
    correct = sum(1 for q, r in zip(exam, results) if q["correct_answer"].startswith(r))
    return correct / len(exam)


# Main function to run the exam
def run_closed_book_exam(model_device: str, model_path: str, model_name: str, task_name: str, exam_file: str):
    exam = load_exam(exam_file)
    
    if model_device == "GCP":
        print("Using transformer")
        model = LlamaGcpModel(
            model_size="70B",
            use_gpu=True,
            model_config=model_config,
            # load_in_4bit=True,
            )
    elif model_device == "claude":
        model = Claude_GCP()
    else:
        print("Using Llama-cpp")
        model = LlamaModel(model_path=model_path)

    results = []
    for question in tqdm(exam, desc="Processing questions", unit="question"):
        if model_device == "GCP":
            answer = generate_answer_llama(model, question["question"], question["choices"])
        else:
            answer = generate_answer(model, question["question"], question["choices"])
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

    with open(f"Data/{task_name}/ExamResults/l3_exam_results_{model_name}_{task_name}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exam completed. Accuracy: {accuracy:.2%}")
    print(f"Results saved to exam_results_{model_name}.json")


if __name__ == "__main__":
    model_device = "GCP"
    # model_device = "claude"
    model_path = "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF"
    model_name = "llama3-70b"
    # model_name = "claude"
    # task_name = "Arxiv"
    # task_name = "LawStackExchange"
    task_name = "StackExchange"
    exam_file = f"Data/{task_name}/ExamData/claude_gcp_2024103016/exam_1000_42.json"
    
    # Create the full directory path
    directory = f"Data/{task_name}/ExamResults"
    os.makedirs(directory, exist_ok=True)
    
    run_closed_book_exam(model_device, model_path, model_name, task_name, exam_file)
