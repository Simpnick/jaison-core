import torch
import torchaudio
import numpy as np
import io
import wave
import importlib.util
from transformers import AutoModel, AutoProcessor
from .base import TTSOperation

class MossTTS(TTSOperation):
    def __init__(self):
        super().__init__("moss")
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "OpenMOSS-Team/MOSS-TTS"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.sampling_rate = 24000
        self.attn_implementation = "auto"
        
    async def start(self) -> None:
        await super().start()
        # Disable the broken cuDNN SDPA backend if it exists in this torch version
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        
        # Prefer FlashAttention 2 when package + device conditions are met.
        if (
            self.device == "cuda" 
            and importlib.util.find_spec("flash_attn") is not None 
            and self.dtype in {torch.float16, torch.bfloat16}
        ):
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                self.attn_implementation = "flash_attention_2"
            else:
                self.attn_implementation = "sdpa"
        elif self.device == "cuda":
            self.attn_implementation = "sdpa"
        else:
            self.attn_implementation = "eager"

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        if hasattr(self.processor, "audio_tokenizer"):
            self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(self.device)
            
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        self.sampling_rate = self.processor.model_config.sampling_rate

    async def close(self) -> None:
        await super().close()
        self.model = None
        self.processor = None

    async def configure(self, config_d):
        if "model_name" in config_d: self.model_name = str(config_d["model_name"])
        if "device" in config_d: self.device = str(config_d["device"])
        
    async def get_configuration(self):
        return {
            "model_name": self.model_name,
            "device": self.device,
            "attn_implementation": self.attn_implementation
        }

    async def _generate(self, content: str = None, **kwargs):
        if not self.model or not self.processor:
            raise RuntimeError("MossTTS is not loaded. Call start() first.")
            
        # Standard MOSS-TTS usage
        conversations = [[self.processor.build_user_message(text=content)]]
        batch = self.processor(conversations, mode="generation")
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4096,
            )
            
        messages = self.processor.decode(outputs)
        if not messages or messages[0] is None:
             raise RuntimeError("The model did not return a decodable audio result.")
             
        audio = messages[0].audio_codes_list[0]
        
        # Convert to bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), self.sampling_rate, format="wav")
        buffer.seek(0)
        
        with wave.open(buffer, "rb") as f:
            sr = f.getframerate()
            sw = f.getsampwidth()
            ch = f.getnchannels()
            ab = f.readframes(f.getnframes())
            
        yield {
            "audio_bytes": ab,
            "sr": sr,
            "sw": sw,
            "ch": ch
        }
