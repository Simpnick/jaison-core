import wave
import os
import logging
from rvc.modules.vc.modules import VC
from pathlib import Path
import torch
import fairseq

from utils.config import Config

from .base import FilterAudioOperation

class RVCFilter(FilterAudioOperation):
    TARGET_SR = 16000
    TARGET_SW = 2
    TARGET_CH = 1
    
    def __init__(self):
        super().__init__("rvc")
        self.vc = None
        
        self.voice: str = None
        self.f0_up_key: int = 0
        self.f0_method: str = "rmvpe"
        self.f0_file: str = None
        self.index_file: str = None
        self.index_rate: float = 0
        self.filter_radius: int = 3
        self.resample_sr: int  = 0
        self.rms_mix_rate: float = 0
        self.protect: float = 0.5
        
        torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
        
    async def start(self):
        # Configurar caminhos para o motor RVC encontrar os modelos base
        os.environ["weight_root"] = os.path.abspath("models/rvc")
        os.environ["hubert_path"] = os.path.abspath("models/rvc/hubert_base.pt")
        os.environ["rmvpe_root"] = os.path.abspath("models/rvc")
        os.environ["index_root"] = os.path.abspath("models/rvc")
        
        await super().start()
        self.vc = VC()
        model_name = self.voice if self.voice.endswith('.pth') else f"{self.voice}.pth"
        
        # Se o model_name não for um caminho absoluto/existente, tenta na pasta de modelos
        if not os.path.exists(model_name):
            model_name = os.path.join("models/rvc", model_name)
            
        self.vc.get_vc(model_name)
        
        # Warmup: Pré-carregar o modelo RMVPE para evitar lag na primeira resposta
        if self.f0_method == "rmvpe":
            import numpy as np
            logging.info("Aquecendo RVC (Carregando RMVPE)...")
            # Faz uma inferência dummy com 1s de silêncio para carregar os modelos com segurança
            self.vc.vc_inference(1, np.zeros(16000, dtype=np.float32), f0_method=self.f0_method)
            logging.info("RVC pronto na agulha!")
    
    async def configure(self, config_d):
        '''Configure and validate operation-specific configuration'''
        if "voice" in config_d: self.voice = str(config_d["voice"])
        if "f0_up_key" in config_d: self.f0_up_key = int(config_d["f0_up_key"])
        if "f0_method" in config_d: self.f0_method = str(config_d["f0_method"])
        if "f0_file" in config_d: self.f0_file = str(config_d["f0_file"])
        if "index_file" in config_d: self.index_file = str(config_d["index_file"])
        if "index_rate" in config_d: self.index_rate = float(config_d["index_rate"])
        if "filter_radius" in config_d: self.filter_radius = int(config_d["filter_radius"])
        if "resample_sr" in config_d: self.resample_sr = int(config_d["resample_sr"])
        if "rms_mix_rate" in config_d: self.rms_mix_rate = float(config_d["rms_mix_rate"])
        if "protect" in config_d: self.protect = float(config_d["protect"])
        
        # TODO check assertions
        
    async def get_configuration(self):
        '''Returns values of configurable fields'''
        return {
            "voice": self.voice,
            "f0_up_key": self.f0_up_key,
            "f0_method": self.f0_method,
            "f0_file": self.f0_file,
            "index_file": self.index_file,
            "index_rate": self.index_rate,
            "filter_radius": self.filter_radius,
            "resample_sr": self.resample_sr,
            "rms_mix_rate": self.rms_mix_rate,
            "protect": self.protect
        }

    async def _generate(self, audio_bytes: bytes = None, sr: int = None, sw: int = None, ch: int = None, **kwargs):
        with wave.open(Config().ffmpeg_working_src, 'wb') as f:
            f.setframerate(sr)
            f.setsampwidth(sw)
            f.setnchannels(ch)
            f.writeframes(audio_bytes)
            
        tgt_sr, audio_opt, times, _ = self.vc.vc_inference(
            1,
            Path(Config().ffmpeg_working_src),
            f0_up_key=self.f0_up_key,
            f0_method=self.f0_method,
            f0_file=self.f0_file,
            index_file=self.index_file,
            index_rate=self.index_rate,
            filter_radius=self.filter_radius,
            resample_sr=self.resample_sr,
            rms_mix_rate=self.rms_mix_rate,
            protect=self.protect,
        )
        
        yield {
            "audio_bytes": audio_opt.tobytes(),
            "sr": tgt_sr,
            "sw": self.TARGET_SW,
            "ch": self.TARGET_CH
        }
    