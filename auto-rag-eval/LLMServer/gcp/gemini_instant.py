import time
from typing import Generator, Dict, Any
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, SafetySetting, HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential

from LLMServer.base_model import BaseLLM

STOP_AFTER_ATTEMPT = 6
WAIT_EXPONENTIAL_MIN = 4
WAIT_EXPONENTIAL_MAX = 30

CONFIG = GenerationConfig(
            temperature=0.0,
            top_p=None,
            top_k=None,
            candidate_count=None,
            max_output_tokens=None,
            stop_sequences=None
            )

SAFETY_CONFIG = [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
    ]


def delayed_text_generator(text: str, delay: float = 0.2):
    tokens = text.split()
    for i in range(1, len(tokens) + 1):
        time.sleep(delay)
        yield " ".join(tokens[:i])

class GeminiGcp(BaseLLM, ):
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro-002",
        project_id: str = "rag-evaluation-437417",
        region: str = "europe-west1"
    ):
        # self.model_name = model_name
        # self.project_id = project_id
        
        vertexai.init(project=project_id, location=region)
        
        self.model = GenerativeModel(model_name=model_name,  generation_config=CONFIG)
        
    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN, max=WAIT_EXPONENTIAL_MAX),
    )
    def invoke(self, prompt: str, params: Dict[str, Any] = None) -> str:
        # inference_params = {**self.inference_params, **(params or {})}

        response = self.model.generate_content(prompt, safety_settings=SAFETY_CONFIG)

        return response.text


    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN, max=WAIT_EXPONENTIAL_MAX),
    )
    def stream_inference(
        self, prompt: str, params: Dict[str, Any] = None
    ) -> Generator[str, None, None]:
        try:
            response = self.model.generate_content(
                prompt,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.inference_params["max_output_tokens"],
                    temperature=self.inference_params["temperature"],
                    top_p=self.inference_params["top_p"],
                )
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    time.sleep(0.2)  # Similar delay as delayed_text_generator
                    
        except Exception as e:
            # Handle streaming-specific errors
            raise e

    def get_id(self) -> str:
        return "gemini_gcp"