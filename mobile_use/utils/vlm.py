import time
import asyncio
import logging
from typing import Any, Optional
from openai import OpenAI, ChatCompletion


logger = logging.getLogger(__name__)


class VLMWrapper:
    """VLM Wrapper for UI Operation.
    """

    def __init__(
            self,
            model_name: str,
            api_key: str,
            base_url: str,
            max_retry: int = 3,
            retry_waiting_seconds: int = 2,
            max_tokens: int = 1024,
            temperature: float = 0.0,
            max_pixels: Optional[int] = None,
            **vlm_kwargs
        ):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # self.aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.max_retry = max_retry
        self.retry_waiting_seconds = retry_waiting_seconds

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_pixels = max_pixels

        self.vlm_kwargs = vlm_kwargs

    def predict(self, messages, stream: bool=False, **kwargs) -> ChatCompletion:
        """Predict the next action given the history and the current screenshot.

        Args:
            messages: The messages to send to the model.
        
        Returns:
            The ChatCompletion of the VLM
        """
        counter = self.max_retry
        wait_seconds = self.retry_waiting_seconds

        # import copy
        # messages_s = copy.deepcopy(messages)
        # for msg in messages_s:
        #     content = msg['content']
        #     if isinstance(content, list):
        #         for cnt in content:
        #             cnt.pop('image_url', None)
        # import json
        # print("messages: ", json.dumps(messages_s, ensure_ascii=False, indent=2))

        # Add max_pixels to all image content if specified
        if self.max_pixels is not None:
            import copy
            messages = copy.deepcopy(messages)
            for msg in messages:
                content = msg.get('content')
                if isinstance(content, list):
                    for cnt in content:
                        if 'image' in cnt.get('type', ''):
                            cnt['max_pixels'] = self.max_pixels

        kwargs.update(self.vlm_kwargs)
        while counter > 0:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=stream,
                    **kwargs
                )
                return response
            except Exception as err:
                logger.warning(f'Error calling VLM API with message: {err}')
                time.sleep(wait_seconds)
                counter -= 1
                if counter <= 0:
                    raise ValueError("Max tries for predict failure.") from err  # re-raise after max retry.
