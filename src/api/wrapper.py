from openai import AsyncOpenAI
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class QAWrapper:
    """Asynchronous wrapper for LLM API client."""

    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1"]

    def __init__(self, model_name: str, api_key: str, max_retries: int = 5):
        """
        Initialize an async API wrapper instance.

        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="http://redservingapi.devops.xiaohongshu.com/v1"
        )

        self.stats = {
            "calls": 0,
            "errors": 0,
            "retries": 0
        }

    async def qa(self, system_prompt: str, user_prompt: str = "", rational: bool = False) -> Any:
        """
        Send a prompt to the model and get a response.

        Args:
            system_prompt: System message
            user_prompt: User query content
            rational: Whether to enable deep reasoning mode

        Returns:
            If rational=True, returns dict with answer and reasoning.
            Otherwise, returns the answer string.

        Raises:
            ValueError: If reasoning is requested but not supported by the model
        """
        if rational and self.model_name not in self.SUPPORTED_REASONING_MODELS:
            raise ValueError(f"Model {self.model_name} does not support reasoning")

        # Retry mechanism with exponential backoff
        for attempt in range(self.max_retries):
            try:
                if rational:
                    return await self._qa_with_reasoning(system_prompt, user_prompt)
                else:
                    return await self._qa_standard(system_prompt, user_prompt)

            except Exception as e:
                self.stats["errors"] += 1
                self.stats["retries"] += 1

                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")

                if attempt == self.max_retries - 1:
                    raise

                # Exponential backoff before retrying
                # retry_delay = 2 ** attempt
                # await asyncio.sleep(retry_delay)

    async def _qa_standard(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        """Execute a standard query without reasoning."""
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ],
            stream=False,
            temperature=1
        )

        self.stats["calls"] += 1
        return {
            "answer": completion.choices[0].message.content,
            "rational": ""
        }

    async def _qa_with_reasoning(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        """Execute a query with reasoning enabled."""
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                },
                {
                    "role": "assistant",
                    "content": "<think>\n"
                }
            ],
            stream=False,
            temperature=1
        )

        self.stats["calls"] += 1
        return {
            "answer": completion.choices[0].message.content,
            "rational": completion.choices[0].message.reasoning_content
        }

    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics for this API instance."""
        return self.stats.copy()