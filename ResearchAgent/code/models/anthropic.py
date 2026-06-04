import os
import time
from typing import Dict, List

from anthropic import Anthropic


class AnthropicClient:
    def __init__(self, model: str = 'deepseek-chat') -> None:
        self._client = Anthropic(
            base_url=os.environ.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
            auth_token=os.environ.get('ANTHROPIC_AUTH_TOKEN', ''),
            timeout=60
        )
        self.model = model

    def call(self, messages: List[Dict[str, str]], max_retries: int = 3, max_tokens: int = 4096) -> str:
        system = None
        user_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                system = msg['content']
            else:
                user_messages.append({'role': msg['role'], 'content': msg['content']})

        attempt = 0
        while True:
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=user_messages,
                    temperature=1.0
                )
                if system:
                    kwargs['system'] = system

                response = self._client.messages.create(**kwargs)

                text_parts = []
                for block in response.content:
                    if hasattr(block, 'text'):
                        text_parts.append(block.text)
                return '\n'.join(text_parts).strip()
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    return str(e)
                time.sleep(4 ** attempt)
                continue
