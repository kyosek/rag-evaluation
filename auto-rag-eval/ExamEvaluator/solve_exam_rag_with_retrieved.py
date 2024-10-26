import json
from typing import List, Dict
from LLMServer.llama.llama_instant import LlamaModel
from tqdm import tqdm


# Load the JSON file
def load_exam(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Generate an answer for a given question
def generate_answer(model: LlamaModel, question: str, choices: List[str], context) -> str:
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{choice}\n"
    prompt += f"\nContext:\n {context}"
    prompt += """
    \nYou are a student that is solving the exam.
    Please solve the question by using the given context and 
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
def run_rag_exam(model_path: str, model_name: str, task_name: str, exam_file: str, retriever: str):
    exam = load_exam(exam_file)
    model = LlamaModel(model_path=model_path)

    results = []
    for question in tqdm(exam, desc="Processing questions", unit="question"):
        try:
            answer = generate_answer(
                model,
                question["question"],
                question["choices"],
                question["retrieved_context"][retriever]
            )
            results.append(answer)
        except:
            pass

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

    with open(f"Data/{task_name}/ExamResults/rag_exam_results_{model_name}_{retriever}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exam completed. Accuracy: {accuracy:.2%}")
    print(f"Results saved to exam_results_{model_name}.json")


if __name__ == "__main__":
    model_path = "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF"
    model_name = "llamav2"
    task_name = "LawStackExchange"
    folder_name = "claude_gcp_2024102123"
    # task_name = "Arxiv"
    # folder_name = "claude_gcp_2024100422"
    exam_file = f"Data/{task_name}/ExamData/{folder_name}/exam_1000_42.json"
    retrievers = ["BM25", "DPR", "SIAMESE", "MultiQA", "DPR:MultiQA:BM25"]
    # retrievers = ["MultiQA", "DPR:MultiQA:BM25"]
    for retriever in retrievers:
        print(f"Retriever: {retriever}")
        run_rag_exam(model_path, model_name, task_name, exam_file, retriever)
