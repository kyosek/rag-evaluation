import json
import time
from typing import Generator, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

class LlamaGcpModel(BaseLLM):
    AVAILABLE_MODELS = {
        "3B": {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8",
        },
        "70B": {
            "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        },
        "Ministral-8B": {
            "model_id": "bartowski/Ministral-8B-Instruct-2410-GGUF"
        }
    }

    def __init__(
        self,
        model_size: str = "3B",
        model_id: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        inference_params: Optional[Dict[str, Any]] = None,
        use_gpu: bool = True,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.float16,
    ):
        """
        Initialize the Llama model with Hugging Face implementation.
        
        Args:
            model_size: Size of the model ("3B", "70B", etc.)
            model_id: Optional custom model ID from Hugging Face
            model_config: Optional model configuration parameters
            inference_params: Optional inference parameters
            use_gpu: Whether to use GPU acceleration
            device_map: Device mapping strategy ("auto", "balanced", "sequential", etc.)
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization
            torch_dtype: Data type for model weights
        """
        # Set default inference parameters
        self.inference_params = {
            "max_new_tokens": 2048,
            "temperature": 0,
            "top_p": 0.9,
            "do_sample": True,
        }
        if inference_params:
            self.inference_params.update(inference_params)

        # Set model ID
        if model_id:
            self.model_id = model_id
        else:
            if model_size not in self.AVAILABLE_MODELS:
                raise ValueError(f"Model size {model_size} not available. Choose from {list(self.AVAILABLE_MODELS.keys())}")
            self.model_id = self.AVAILABLE_MODELS[model_size]["model_id"]
        
        self.model_size = model_size

        try:
            # Configure quantization
            quantization_config = None
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )

            # Set device configuration
            if use_gpu and torch.cuda.is_available():
                device = "cuda"
                device_map = device_map
            elif use_gpu and torch.cuda.is_available():
                device = "mps"
                device_map = device_map
            else:
                device = "cpu"
                device_map = None
                torch_dtype = torch.float32

            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                **model_config if model_config else {}
            )

            # Log configuration
            print(f"Model initialized with configuration:")
            print(f"- Model: {self.model_id}")
            print(f"- Device: {device}")
            print(f"- Device Map: {device_map}")
            print(f"- Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'None'}")
            print(f"- Torch dtype: {torch_dtype}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _format_chat_prompt(self, prompt: str) -> str:
        """Format the prompt for chat completion."""
        return f"[INST] {prompt} [/INST]"

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN, max=WAIT_EXPONENTIAL_MAX),
    )
    def invoke(self, prompt: str) -> str:
        try:
            formatted_prompt = self._format_chat_prompt(prompt)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.inference_params
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the response
            response = response[len(formatted_prompt):].strip()
            return response
        except Exception as e:
            raise ValueError(f"Incorrect Generation: {str(e)}")

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN, max=WAIT_EXPONENTIAL_MAX),
    )
    def stream_inference(self, prompt: str) -> Generator[str, None, None]:
        try:
            formatted_prompt = self._format_chat_prompt(prompt)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # Get the length of input tokens to skip them in the output
            input_length = inputs["input_ids"].shape[1]
            
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.inference_params
            )

            # Create a thread to run the generation
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Yield generated text chunks
            for chunk in streamer:
                yield chunk

        except Exception as e:
            raise ValueError(f"Incorrect Generation: {str(e)}")

    def get_id(self):
        return f"LlamaModel:2-{self.model_size}"
