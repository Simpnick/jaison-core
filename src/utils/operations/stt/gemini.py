from google import genai
from google.genai import types
import os
import wave
import logging
from utils.config import Config
from .base import STTOperation

class GeminiSTT(STTOperation):
    def __init__(self):
        super().__init__("gemini")
        self.model_name = "gemini-2.0-flash"
        self.language = "pt-BR"
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
        if "language" in config_d: self.language = str(config_d['language'])

        assert self.model_name is not None and len(self.model_name) > 0

    async def get_configuration(self):
        '''Returns values of configurable fields'''
        return {
            "model": self.model_name,
            "language": self.language
        }

    async def _generate(self, prompt: str = None, audio_bytes: bytes = None, sr: int = None, sw: int = None, ch: int = None, **kwargs):
        # The new SDK's generate_content with audio
        # We need to send the audio bytes and a prompt to transcribe
        
        # Gemini works best with a prompt like "Transcribe the following audio"
        # We can also include the language hint
        transcription_prompt = f"Transcreva o áudio a seguir exatamente como falado. Idioma: {self.language}."
        if prompt:
            transcription_prompt += f" Contexto adicional: {prompt}"

        # Create audio part
        # Note: We use the raw bytes if they are in a format Gemini likes (wav/mp3/etc)
        # JAIson usually provides raw PCM bytes that need to be wrapped in WAVE
        
        import io
        byte_io = io.BytesIO()
        with wave.open(byte_io, 'wb') as f:
            f.setframerate(sr)
            f.setsampwidth(sw)
            f.setnchannels(ch)
            f.writeframes(audio_bytes)
        
        wav_bytes = byte_io.getvalue()

        # Call Gemini (async)
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                transcription_prompt
            ]
        )

        transcription = response.text.strip() if response.text else ""
        
        yield {"transcription": transcription}
