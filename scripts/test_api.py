import asyncio
import json
import yaml
import logging
from api import StreamGenerator
from utils import setup_logger
from typing import List

async def generate_to_file(
        prompts: List[str],
        output_file: str,
        model_name: str,
        api_keys: List[str],
        system_prompt: str = "",
        max_concurrent_per_key: int = 300,
        max_retries: int = 5,
):
    """
    Generate responses for prompts and stream results to a JSONL file.

    Args:
        prompts: List of prompts to process
        output_file: Path to output JSONL file
        model_name: Name of model to use
        api_keys: List of API keys
        system_prompt: System prompt for all requests
        max_concurrent_per_key: Max concurrent requests per key
        max_retries: Max retries per request
        rational: Whether to enable reasoning
    """
    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1"]

    generator = StreamGenerator(
        model_name=model_name,
        api_keys=api_keys,
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
        rational=model_name in SUPPORTED_REASONING_MODELS,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        # validate_func(answer:str)->bool
        async for result in generator.generate_stream(prompts, system_prompt, validate_func=None):
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()  # Ensure immediate write


# Example usage
if __name__ == "__main__":
    logger = setup_logger(logging.INFO)
    logger.info("Starting !")

    prompts = ["Tell me about Paris", "Describe Tokyo", "Explain quantum computing"]
    api_keys = yaml.safe_load(open("./configs/keys.yaml", "r")).get("keys", [])

    asyncio.run(
        generate_to_file(
            prompts=prompts,
            output_file="../output.jsonl",
            model_name="deepseek-v3-0324",
            api_keys=api_keys,
            system_prompt=""
        )
    )