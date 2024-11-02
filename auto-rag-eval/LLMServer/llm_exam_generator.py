import logging
from os.path import abspath, dirname
from typing import Dict, List

from tqdm import tqdm
from LLMServer.base_model import BaseLLM

logger = logging.getLogger(__name__)
ROOTPATH = dirname(dirname(abspath(__file__)))


class LLMExamGenerator:

    def __init__(self, step_size: int, task_domain: str, llm_model: BaseLLM):

        # Step size is to mitigate when one model inference is faster than another
        # eg openllama:13b = 3* llamav2:70B
        self.step_size = step_size
        self.task_domain = task_domain
        self.llm_model = llm_model

    def make_question_prompt(self, documentation: str) -> str:
        # Adding the syntax constraint was done in V2 and appears to impact the formatting of the question.
        return (
            f"### Human: Here is some documentation from {self.task_domain}: {documentation}.\n"
            "From this generate a difficult multi-form question for an exam. It should have 4 candidates,"
            " 1 correct answer and explanations. Syntax should be Question: {question}\nA){candidate A}\nB){candidate B}\n"
            "C){candidate C}\nD){candidate D} Correct Answer: {correct answer}\n### Assistant:"
        )

    # def make_question_prompt_icl(self, example, documentation: str) -> str:
    #     # icl = (f"### Human: Here is some documentation from {self.task_domain}: {example.documentation}.\n"
    #     #        f"From this generate a difficult multi-form question for an exam. It should have 4 candidates,"
    #     #        " 1 correct answer and explanations.\n### Assistant:"
    #     #        "Question: {}\nCandidates: {}\n".format(example.question, '\n'.join(example.choices))
    #     #        f"Correct Answer: {example.correct_answer}\n")
    #     prompt = (f"### Human: Here is some documentation from {self.task_domain}: {documentation}.\n"
    #               f"From this generate a difficult multi-form question for an exam. It should have 4 candidates,"
    #               " 1 correct answer and explanations.\n### Assistant:")
    #     return f"{icl}\n{prompt}"

    def generate_exam(self, data: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:

        generated_questions = {}
        for k in tqdm(range(0, len(data), self.step_size)):
            answer = self.llm_model.invoke(
                prompt=self.make_question_prompt(data[k]["text"]), params={}
            )
            generated_questions[k] = {"documentation": data[k], "answer": answer}
        return generated_questions


class ClaudeExamGenerator(LLMExamGenerator):

    def __init__(self, step_size: int, task_domain: str, llm_model: BaseLLM):

        super().__init__(step_size=step_size, task_domain=task_domain, llm_model=llm_model)

    def make_question_prompt(self, documentation: str) -> str:
        return (
            f"\n\nHuman: Here is some documentation from {self.task_domain}: {documentation}.\n"
            "From this generate a difficult multi-form question for an exam. It should have 4 candidates,"
            " 1 correct answer and explanations. Syntax should be Question: {question}\nA){candidate A}\nB){candidate B}\n"
            "C){candidate C}\nD){candidate D} Correct Answer: {correct answer}\n\nAssistant:"
        )

    def generate_exam(self, data: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:

        generated_questions = {}
        for k in tqdm(range(0, len(data), self.step_size)):
            answer = self.llm_model.invoke(
                prompt=self.make_question_prompt(data[k]["text"]), params={}
            )
            generated_questions[k] = {"documentation": data[k], "answer": answer}
        return generated_questions


class LlamaExamGenerator(LLMExamGenerator):

    def __init__(self, step_size: int, task_domain: str, llm_model: BaseLLM):

        super().__init__(step_size=step_size, task_domain=task_domain, llm_model=llm_model)

    def make_question_prompt(self, documentation: str) -> str:
        return (
            f"\n\nHuman: Here is some documentation from {self.task_domain}: {documentation}.\n"
            "From this generate a difficult multi-form question for an exam. It should have 4 candidates,"
            " 1 correct answer and explanations. Syntax should be Question: {question}\nA){candidate A}\n"
            "B){candidate B}\nC){candidate C}\nD){candidate D} Correct Answer: {correct answer}\n\nAssistant:"
        )
        
    def make_l3_question_prompt(self, documentation: str) -> str:
        return (
            f"""\n\n
            <<SYS>>
            You are an AI assistant to generate a difficult multiple choice question that are
            necessitate external data to furnish rationales for its resolution.
            The question demands not only a grasp of the factual content but also 
            the ability to comprehend and apply domain-specific rationales that are
            integral to the data’s context.
            Your task is:
                1. Read and comprehend the given document in a specific domain
                2. Come up with a question that is described the above
                3. Generate a multiple-choice question with 4 candidates: 1 correct answer
                4. Follow the output instructions 
            <</SYS>>
            
            Here is a documentation from domain: {self.task_domain}:\n
            Document: {documentation}\n

            Output instructions:
                - It must have 4 candidates
                - 1 correct answer 
                - The output must be:
                    Question: {question}\n
                    A) candidate A\n
                    B) candidate B\n
                    C) candidate C\n
                    D) candidate D\n
                    Correct Answer: correct answer

            Example of questions:
                - How should a patient with chest pain and specific symptom descriptions be diagnosed and treated *(given a chest pain management guideline)
                - How to respond to a user’s question in a real-life scenario? *(given a customer service workflow)

            Your answer:
            """
        )

    def generate_exam(self, data: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:

        generated_questions = {}
        for k in tqdm(range(0, len(data), self.step_size)):
            answer = self.llm_model.invoke(
                # prompt=self.make_question_prompt(data[k]["text"]),
                prompt=self.make_l3_question_prompt(data[k]["text"]),
                # params={}
            )
            generated_questions[k] = {"documentation": data[k], "answer": answer}
        return generated_questions
