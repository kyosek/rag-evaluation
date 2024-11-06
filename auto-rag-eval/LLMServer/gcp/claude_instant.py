import time
from typing import Generator, Dict, Any

from google.cloud import aiplatform
from anthropic import AnthropicVertex
from LLMServer.base_model import BaseLLM
from tenacity import retry, stop_after_attempt, wait_exponential

STOP_AFTER_ATTEMPT = 6
WAIT_EXPONENTIAL_MIN = 4
WAIT_EXPONENTIAL_MAX = 30


def delayed_text_generator(text: str, delay: float = 0.2):
    tokens = text.split()
    for i in range(1, len(tokens) + 1):
        time.sleep(delay)
        yield " ".join(tokens[:i])


class ClaudeGcp(BaseLLM):

    def __init__(self):
        self.client = AnthropicVertex(project_id="rag-evaluation-437417", region="europe-west1")
        # aiplatform.init(project='rag-evaluation-437417', location='us-central1')
        # self.endpoint = aiplatform.Endpoint('projects/rag-evaluation-437417/locations/us-central1/endpoints/rag-evaluation-437417')
        self.inference_params = {
            "max_tokens_to_sample": 4096,
            "temperature": 0,
            "top_p": 0.9,
        }

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN, max=WAIT_EXPONENTIAL_MAX),
    )
    def invoke(self, prompt: str, params: Dict[str, Any] = None) -> str:
        inference_params = {**self.inference_params, **(params or {})}

        # Prepare the instance
        instance = {"prompt": prompt, **inference_params}

        response = self.client.messages.create(
            model="claude-3-5-sonnet@20240620",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ],
        )
        # response = self.endpoint.predict([instance])

        if response.content[0].text:
            return response.content[0].text

        raise ValueError("Incorrect Generation")

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN, max=WAIT_EXPONENTIAL_MAX),
    )
    def stream_inference(
        self, prompt: str, params: Dict[str, Any] = None
    ) -> Generator[str, None, None]:
        completion = self.invoke(prompt, params)
        return delayed_text_generator(completion)

    def get_id(self) -> str:
        return "claude_gcp"
