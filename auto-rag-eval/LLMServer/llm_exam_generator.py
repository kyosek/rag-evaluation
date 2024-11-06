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
            f"\n\nHuman: Here is some documentation from {self.task_domain}:\n\n{documentation}\n\n"
            "Generate a challenging multiple-choice question that tests deep understanding of the material. "
            "The question should:\n"
            "1. Test application of concepts rather than mere recall\n"
            "2. Include plausible distractors that represent common misconceptions\n"
            "3. Have clear, unambiguous wording\n\n"
            "Format:\n"
            "Question: [Question text]\n"
            "A) [First option]\n"
            "B) [Second option]\n"
            "C) [Third option]\n"
            "D) [Fourth option]\n"
            "Correct Answer: [Letter of correct answer]\n"
            "Explanation: [Detailed explanation of why the correct answer is right and why others are wrong]\n\n"
            "Assistant:"
        )
    
    def make_l3_question_prompt(self, documentation: str) -> str:
        return f"""
        <<SYS>>
        You are an expert exam question generator specializing in creating high-quality, challenging multiple-choice questions. 
        Your questions should:
        1. Target L3 (Analysis/Application) or higher cognitive levels in Bloom's taxonomy
        2. Require integration of multiple concepts from the documentation
        3. Include real-world applications or scenarios
        4. Test critical thinking rather than memorization
        5. Have carefully crafted distractors that represent common misconceptions

        Guidelines for creating options:
        - All options should be of similar length and complexity
        - Avoid obvious wrong answers
        - Use common misconceptions as distractors
        - Ensure options are mutually exclusive
        - Avoid "all/none of the above" options
        <</SYS>>

        Domain: {self.task_domain}
        Documentation: {documentation}

        Example questions based on different domains:

        Technical Documentation Example:
        Question: A microservice is experiencing intermittent failures during peak load. Given the following error logs and system metrics, what is the most likely root cause?
        A) Network timeout due to connection pool exhaustion
        B) Memory leak in the application container
        C) Database connection throttling
        D) CPU throttling by the container orchestrator
        Correct Answer: A
        Explanation: The logs show increasing connection wait times...

        Medical Guidelines Example:
        Question: A 45-year-old patient presents with acute chest pain (7/10), radiating to the left arm, with associated shortness of breath. The ECG shows ST-segment elevation in leads V1-V4. Based on the current guidelines, what is the most appropriate immediate management?
        A) Administer aspirin and arrange urgent PCI
        B) Start thrombolysis and transfer to nearest cardiac center
        C) Perform bedside echocardiogram before any intervention
        D) Administer morphine and arrange CT coronary angiogram
        Correct Answer: A
        Explanation: Given the STEMI presentation...

        Please generate a question following this format:
        Question: [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        Correct Answer: [Letter]
        Explanation: [Detailed explanation]
        """
    
    def generate_exam(self, data: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:

        generated_questions = {}
        for k in tqdm(range(0, len(data), self.step_size)):
            answer = self.llm_model.invoke(
                prompt=self.make_l3_question_prompt(data[k]["text"]), params={}
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
                    Question: question\n
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
