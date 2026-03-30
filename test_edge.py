import asyncio
from src.utils.operations.tts.edge import EdgeTTS
from pydub import AudioSegment
from io import BytesIO
import winsound

async def test_edge():
    print("Iniciando teste do Edge TTS...")
    tts = EdgeTTS()
    
    # Configuração
    await tts.configure({
        "voice": "pt-BR-FranciscaNeural"
    })
    
    text = "Oi! Eu sou a Francisca do Edge TTS. O que você achou da minha voz em comparação com a anterior?"
    print(f"Gerando áudio para: {text}")
    
    async for chunk in tts._generate(content=text):
        audio_bytes = chunk["audio_bytes"]
        sr = chunk["sr"]
        sw = chunk["sw"]
        ch = chunk["ch"]
        
        print(f"Áudio gerado: {len(audio_bytes)} bytes, SR: {sr}, SW: {sw}, CH: {ch}")
        
        # Converter para WAV completo em memória
        audio = AudioSegment(
            data=audio_bytes,
            sample_width=sw,
            frame_rate=sr,
            channels=ch
        )
        
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        
        print("Reproduzindo...")
        winsound.PlaySound(wav_io.getvalue(), winsound.SND_MEMORY)

if __name__ == "__main__":
    asyncio.run(test_edge())
