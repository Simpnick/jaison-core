from google import genai
import os
import struct
import base64
from .base import EmbeddingOperation

class GeminiEmbedding(EmbeddingOperation):
    def __init__(self):
        super().__init__("gemini")
        self.model_name = "models/gemini-embedding-2-preview"
        self.task_type = "RETRIEVAL_DOCUMENT"
        self.client = None

    async def start(self):
        await super().start()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=api_key)

    async def close(self):
        await super().close()
        self.client = None

    async def configure(self, config_d):
        '''Configure and validate operation-specific configuration'''
        if "model" in config_d: self.model_name = str(config_d['model'])
        if "task_type" in config_d: self.task_type = str(config_d['task_type'])

        assert self.model_name is not None and len(self.model_name) > 0

    async def get_configuration(self):
        '''Returns values of configurable fields'''
        return {
            "model": self.model_name,
            "task_type": self.task_type,
        }

    async def _generate(self, content: str = None, **kwargs):
        # The new SDK's embed_content
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=content,
            config={
                'task_type': self.task_type
            }
        )

        # In the new SDK, the result is an object with an 'embeddings' attribute (list) or 'embedding' (single)
        # Assuming single content:
        float_list = result.embeddings[0].values if isinstance(result.embeddings, list) else result.embeddings.values
        
        # Pack floats into bytes (little-endian)
        format_string = '<' + 'f' * len(float_list)
        packed_bytes = struct.pack(format_string, *float_list)
        
        # Encode to base64
        encoded_result = base64.b64encode(packed_bytes).decode('utf-8')

        yield {
            "embedding": encoded_result
        }
