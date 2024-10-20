import json
import time
from typing import Generator
from llama_cpp import Llama
from LLMServer.base_model import BaseLLM
from tenacity import retry, stop_after_attempt, wait_exponential

STOP_AFTER_ATTEMPT = 10
WAIT_EXPONENTIAL_MIN = 4
WAIT_EXPONENTIAL_MAX = 30


def delayed_text_generator(text: str, delay: float = 0.2):
    tokens = text.split()
    for i in range(1, len(tokens) + 1):
        time.sleep(delay)
        yield " ".join(tokens[:i])


class LlamaModel(BaseLLM):
    def __init__(
        self,
        model_path: str = "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
        filename: str = "*q8_0.gguf",
    ):
        self.llm = Llama.from_pretrained(
            repo_id=model_path,
            filename=filename,
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=True,
        )
        self.inference_params = {
            "max_tokens": 8192,
            "temperature": 0,
            "top_p": 0.9,
        }

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN, max=WAIT_EXPONENTIAL_MAX),
    )
    def invoke(self, prompt: str) -> str:
        try:
            output = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}], **self.inference_params
            )
            return output["choices"][0]["message"]["content"]
        except Exception as e:
            raise ValueError(f"Incorrect Generation: {str(e)}")

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN, max=WAIT_EXPONENTIAL_MAX),
    )
    def stream_inference(self, prompt: str) -> Generator[str, None, None]:
        try:
            output = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}], **self.inference_params, stream=True
            )
            full_response = ""
            for chunk in output:
                if chunk["choices"][0]["delta"].get("content"):
                    full_response += chunk["choices"][0]["delta"]["content"]
                    yield chunk["choices"][0]["delta"]["content"]
        except Exception as e:
            raise ValueError(f"Incorrect Generation: {str(e)}")

    def get_id(self):
        return "LlamaModel:3.2-3B"
