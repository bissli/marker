import base64
import json
import os
from io import BytesIO
from typing import Annotated

import PIL
import requests
from marker.schema.blocks import Block
from marker.services import BaseService
from pydantic import BaseModel


class ClaudeService(BaseService):
    api_base: Annotated[str, 'The base url for OpenRouter API'] = os.getenv('OPENROUTER_BASE_URL')
    api_key: Annotated[str, 'Your OpenRouter API key'] = os.getenv('OPENROUTER_API_KEY')
    model: Annotated[str, 'The model to use'] = 'anthropic/claude-3.5-sonnet'

    def image_to_base64(self, image: PIL.Image.Image):
        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        return base64.b64encode(image_bytes.getvalue()).decode('utf-8')

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | list[PIL.Image.Image],
        block: Block,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None
    ):
        url = f'{self.api_base}/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'HTTP-Referer': 'https://localhost:8000',
            'X-Title': 'libb-nlp'
        }

        if not isinstance(image, list):
            image = [image]

        image_bytes = [self.image_to_base64(img) for img in image]

        # Format the messages for Claude
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]

        # Add images to the first message's content
        for img in image_bytes:
            messages[0]['content'].append({
                'type': 'image',
                'image': img
            })

        schema = response_schema.model_json_schema()
        system_prompt = f'You must respond with valid JSON matching this schema: {json.dumps(schema)}'

        payload = {
            'model': self.model,
            'messages': messages,
            'system': system_prompt,
            'stream': False,
            'response_format': {'type': 'json_object'}
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            total_tokens = response_data.get('usage', {}).get('total_tokens', 0)
            block.update_metadata(llm_request_count=1, llm_tokens_used=total_tokens)

            return json.loads(response_data['choices'][0]['message']['content'])
        except Exception as e:
            print(f'OpenRouter inference failed: {e}')

        return {}
