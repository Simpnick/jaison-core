import asyncio
import os
from dotenv import load_dotenv
import wave
import numpy as np
from src.utils.operations.stt.gemini import GeminiSTT

async def verify_stt():
    load_dotenv()
    print("Iniciando verificação do Gemini STT...")
    
    stt = GeminiSTT()
    # Usando gemini-2.0-flash para transcrição rápida
    await stt.configure({"model": "gemini-2.0-flash", "language": "pt-BR"})
    await stt.start()
    
    # Criar um áudio de silêncio ou um tom simples para teste (apenas para validar a API)
    # Na vida real, enviaríamos voz
    duration = 2 # seconds
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    # Silêncio ou ruído branco leve
    audio_data = (np.random.normal(0, 0.01, t.shape) * 32767).astype(np.int16)
    audio_bytes = audio_data.tobytes()
    
    print("Enviando áudio para o Gemini...")
    try:
        async for chunk in stt._generate(
            prompt="Isto é um teste de transcrição. Diga 'Silêncio' se não ouvir nada.",
            audio_bytes=audio_bytes,
            sr=sr,
            sw=2,
            ch=1
        ):
            print(f"Transcrição recebida: '{chunk.get('transcription')}'")
    except Exception as e:
        print(f"Erro na transcrição: {e}")
    finally:
        await stt.close()

if __name__ == "__main__":
    asyncio.run(verify_stt())
