import time
from typing import Generator, Dict, Any

from anthropic import AnthropicVertex
from tenacity import retry, stop_after_attempt, wait_exponential

from LLMServer.base_model import BaseLLM

STOP_AFTER_ATTEMPT = 6
WAIT_EXPONENTIAL_MIN = 4
WAIT_EXPONENTIAL_MAX = 30

# Model region mapping
MODEL_REGION_MAPPING = {
    "claude-3-5-haiku@20241022": "us-east5",
    "claude-3-opus@20240229": "us-east5",
    "claude-3-5-sonnet@20240620": "europe-west1",
}


def delayed_text_generator(text: str, delay: float = 0.2):
    tokens = text.split()
    for i in range(1, len(tokens) + 1):
        time.sleep(delay)
        yield " ".join(tokens[:i])


class ClaudeGcp(BaseLLM):
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet@20240620",
        project_id: str = "rag-evaluation-437417",
        region: str = None,
    ):
        # Determine the appropriate region for the model
        self.model_name = model_name
        self.project_id = project_id

        # If region is not specified, use the region from the mapping
        if region is None:
            self.region = MODEL_REGION_MAPPING.get(
                model_name, "europe-west1"  # Default to europe-west1 if model not found in mapping
            )
        else:
            self.region = region

        # Initialize the client with the appropriate region
        self.client = AnthropicVertex(project_id=self.project_id, region=self.region)

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

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    }
                ],
            )

            if response.content[0].text:
                return response.content[0].text

        except Exception as e:
            if "Model is not available in this region" in str(e):
                suggested_region = MODEL_REGION_MAPPING.get(self.model_name)
                raise ValueError(
                    f"Model {self.model_name} is not available in region {self.region}. "
                    f"Please try using region: {suggested_region}"
                )
            raise e

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
