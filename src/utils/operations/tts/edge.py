import edge_tts
from io import BytesIO
from pydub import AudioSegment
import logging

from .base import TTSOperation

class EdgeTTS(TTSOperation):
    def __init__(self):
        super().__init__("edge")
        self.voice = "pt-BR-FranciscaNeural"
        self.rate = "+0%"
        self.volume = "+0%"
        self.pitch = "+0Hz"

    async def start(self) -> None:
        await super().start()

    async def close(self) -> None:
        await super().close()

    async def configure(self, config_d):
        '''Configure and validate operation-specific configuration'''
        if "voice" in config_d: self.voice = str(config_d["voice"])
        if "rate" in config_d: self.rate = str(config_d["rate"])
        if "volume" in config_d: self.volume = str(config_d["volume"])
        if "pitch" in config_d: self.pitch = str(config_d["pitch"])

    async def get_configuration(self):
        '''Returns values of configurable fields'''
        return {
            "voice": self.voice,
            "rate": self.rate,
            "volume": self.volume,
            "pitch": self.pitch
        }

    async def _generate(self, content: str = None, **kwargs):
        '''Generate output stream using edge-tts'''
        try:
            communicate = edge_tts.Communicate(
                text=content, 
                voice=self.voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch
            )
            
            mp3_buffer = BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.write(chunk["data"])
            
            mp3_buffer.seek(0)
            
            # Converter MP3 para PCM usando pydub
            audio = AudioSegment.from_file(mp3_buffer, format="mp3")
            
            yield {
                "audio_bytes": audio.raw_data,
                "sr": audio.frame_rate,
                "sw": audio.sample_width,
                "ch": audio.channels
            }
        except Exception as e:
            logging.error(f"Erro no EdgeTTS: {e}")
            raise e
