from google import genai
from google.genai import types
import os
import logging
from .base import T2TOperation
from utils.prompter.message import ChatMessage
from utils.prompter import Prompter

class GeminiT2T(T2TOperation):
    def __init__(self):
        super().__init__("gemini")
        self.model_name = "gemini-1.5-flash" # More common standard model
        self.temperature = 1.0
        self.top_p = 0.95
        self.top_k = 40
        self.max_output_tokens = 8192
        self.client = None

    async def start(self):
        await super().start()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        # Use AsyncClient for better performance in JAIson
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    async def close(self):
        await super().close()
        self.client = None

    async def configure(self, config_d):
        '''Configure and validate operation-specific configuration'''
        logging.info(f"Configuring GeminiT2T with: {config_d}")
        if "model" in config_d: 
            self.model_name = str(config_d['model'])
        
        if "temperature" in config_d: self.temperature = float(config_d['temperature'])
        if "top_p" in config_d: self.top_p = float(config_d['top_p'])
        if "top_k" in config_d: self.top_k = int(config_d['top_k'])
        if "max_output_tokens" in config_d: self.max_output_tokens = int(config_d['max_output_tokens'])

        logging.info(f"GeminiT2T configured to use model: {self.model_name}")
        assert self.model_name is not None and len(self.model_name) > 0

    async def get_configuration(self):
        '''Returns values of configurable fields'''
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }

    async def _generate(self, instruction_prompt: str = None, messages: list = None, **kwargs):
        # Format history for Gemini
        history = []
        if messages is None:
            messages = []
            
        for msg in messages:
            role = "user"
            content = ""
            if isinstance(msg, ChatMessage):
                if msg.user == Prompter().character_name:
                    role = "model"
                content = msg.message
            else:
                content = str(msg)
            
            history.append(types.Content(role=role, parts=[types.Part(text=content)]))

        # Config
        config_kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }

        # Inject emotion instructions for Gigi
        emotion_instruction = (
            "\nIMPORTANTE: No final da sua resposta, inclua obrigatoriamente a emocao atual entre tags de XML. "
            "Escolha uma destas etiquetas: [admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, anger, annoyance, disappointment, disapproval, embarrassment, fear, disgust, grief, nervousness, remorse, sadness, confusion, curiosity, realization, relief, surprise, neutral]. "
            "Exemplo: <emotion>joy</emotion>"
        )
        
        if instruction_prompt:
            instruction_prompt += emotion_instruction
        else:
            instruction_prompt = emotion_instruction

        # Some models (like Gemma) don't support system_instruction parameter.
        if "gemma" not in self.model_name.lower():
            config_kwargs["system_instruction"] = instruction_prompt
        else:
            system_msg = types.Content(role="user", parts=[types.Part(text=f"[SYSTEM]: {instruction_prompt}")])
            history.insert(0, system_msg)

        config = types.GenerateContentConfig(**config_kwargs)

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=history,
            config=config
        )

        async for chunk in response_stream:
            if chunk.text:
                yield {"content": chunk.text}
