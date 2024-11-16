import os
import json
import time
from typing import Generator
from llama_cpp import Llama
from enum import Enum
from abc import ABC

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


class ModelType(Enum):
    MISTRAL_7B = "mistral-7b"
    MIXTRAL_8_7B = "mixtral-8-7b"
    MIXTRAL_8_22B = "mixtral-8-22b"
    LLAMA_3_2 = "llama-3-2"


class BaseQuantizedModel(ABC):
    def __init__(
        self,
        model_path: str,
        filename: str,
        n_ctx: int = 16384,
        n_gpu_layers: int = -1,
        verbose: bool = True
    ):
        self.llm = Llama.from_pretrained(
            repo_id=model_path,
            filename=filename,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        self.inference_params = {
            "max_tokens": n_ctx,
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
                messages=[{"role": "user", "content": prompt}],
                **self.inference_params
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
                messages=[{"role": "user", "content": prompt}],
                **self.inference_params,
                stream=True
            )
            full_response = ""
            for chunk in output:
                if chunk["choices"][0]["delta"].get("content"):
                    full_response += chunk["choices"][0]["delta"]["content"]
                    yield chunk["choices"][0]["delta"]["content"]
        except Exception as e:
            raise ValueError(f"Incorrect Generation: {str(e)}")

class Mistral7BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx: int = 16384
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Mistral7BModel:7B"

class Mixtral8x7BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        filename: str = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx: int = 32768
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )


class Mixtral8x22BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "bartowski/Mixtral-8x22B-v0.1-GGUF",
        filename: str = "Mixtral-8x22B-v0.1-IQ4_XS-00001-of-00005.gguf",
        n_ctx: int = 32768
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )


    def get_id(self):
        return "Mixtral8x7BModel:8x7B"

class LlamaModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "hugging-quants/Llama-3.2-3B-Instruct-GGUF",
        filename: str = "*q8_0.gguf",
        n_ctx: int = 16384
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "LlamaModel:3.2-3B"

class ModelFactory:
    @staticmethod
    def create_model(model_type: ModelType, **kwargs) -> BaseQuantizedModel:
        model_map = {
            ModelType.MISTRAL_7B: Mistral7BModel,
            ModelType.MIXTRAL_8_7B: Mixtral8x7BModel,
            ModelType.MIXTRAL_8_22B: Mixtral8x22BModel,
            ModelType.LLAMA_3_2: LlamaModel
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return model_map[model_type](**kwargs)
