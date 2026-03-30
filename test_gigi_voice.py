import asyncio
import os
import torch
import numpy as np
from src.utils.operations.tts.edge import EdgeTTS
from src.utils.operations.filter_audio.rvc import RVCFilter
from pydub import AudioSegment
from io import BytesIO
import winsound

async def test_full_pipeline():
    print("--- Teste Completo: Edge TTS + Filtro RVC (Gigi) ---")
    
    # 1. Configurar TTS (Edge)
    print("Iniciando Edge TTS...")
    tts = EdgeTTS()
    await tts.configure({
        "voice": "pt-BR-FranciscaNeural"
    })
    await tts.start()
    
    # 2. Configurar Filtro (RVC)
    print("Iniciando Filtro RVC com modelo Gigi...")
    rvc = RVCFilter()
    await rvc.configure({
        "voice": "Gigi",
        "index_file": "models/rvc/Gigi.index",
        "f0_up_key": 0,
        "f0_method": "rmvpe",
        "index_rate": 0.75
    })
    await rvc.start()
    
    text = "Oi! Eu sou a Gigi a Sapeca. Agora estou usando o motor do Edge TTS com a minha própria conversão de voz."
    print(f"Gerando áudio base e aplicando filtro...")
    
    # Gerar e filtrar
    all_filtered_audio = b""
    sr, sw, ch = 16000, 2, 1
    
    async for tts_chunk in tts._generate(content=text):
        print(f"Base gerada ({len(tts_chunk['audio_bytes'])} bytes). Filtrando...")
        
        # O filtro RVC espera um stream
        async for filtered_chunk in rvc._generate(
            audio_bytes=tts_chunk["audio_bytes"],
            sr=tts_chunk["sr"],
            sw=tts_chunk["sw"],
            ch=tts_chunk["ch"]
        ):
            all_filtered_audio += filtered_chunk["audio_bytes"]
            sr = filtered_chunk["sr"]
            sw = filtered_chunk["sw"]
            ch = filtered_chunk["ch"]

    if not all_filtered_audio:
        print("Erro: Nenhum áudio foi gerado.")
        return

    print(f"Processamento concluído. Tamanho final: {len(all_filtered_audio)} bytes.")
    
    # Reproduzir
    audio = AudioSegment(
        data=all_filtered_audio,
        sample_width=sw,
        frame_rate=sr,
        channels=ch
    )
    
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    
    print("Reproduzindo resultado final...")
    winsound.PlaySound(wav_io.getvalue(), winsound.SND_MEMORY)

if __name__ == "__main__":
    # Garantir que estamos no diretório correto para caminhos relativos
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    asyncio.run(test_full_pipeline())
