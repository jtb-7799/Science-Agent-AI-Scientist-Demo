import os
import time
from typing import Dict, List

from openai import AzureOpenAI


class OpenAIClient:
    def __init__(self, model: str = 'gpt-4o') -> None:
        self._client = AzureOpenAI(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY', ''),
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT', ''),
        ).with_options(timeout=30)
        self.model = model

    def call(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        attempt = 0

        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=1.25
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    return str(e)

                sleep_s = 4 ** attempt
                time.sleep(sleep_s)
                continue
