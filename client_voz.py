"""
Cliente de Chat por Voz - Gigi a Sapeca
========================================
Push-To-Talk: Segure a tecla [B] para falar, solte para enviar.
Digite 'sair' para encerrar.
"""

import asyncio
import threading
import wave
import json
import base64
import re
import time
import winsound
import requests
import websockets
import pyaudio
import keyboard
import io
import os
from datetime import datetime
import numpy as np

# ──────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────
SERVER_URL     = "http://127.0.0.1:7272"
WS_URL         = "ws://127.0.0.1:7272/"
PTT_KEY        = "b"          # Tecla de Push-To-Talk
SAMPLE_RATE    = 16000
CHANNELS       = 1
CHUNK          = 1024
FORMAT         = pyaudio.paInt16
USER_NAME      = "Usuario"    # Nome que aparece no histórico

# ──────────────────────────────────────────────
# Estado Global
# ──────────────────────────────────────────────
is_recording   = False
stop_ws        = False
audio_queue    = asyncio.Queue()
active_stream  = None
stream_sr      = 0
pa_ctx         = None

# ──────────────────────────────────────────────
# Funções Auxiliares
# ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove tags XML, emotes e emojis — só o texto limpo."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[\[emote:[^\]]+\]\]', '', text)
    text = re.sub(
        u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF\U00002500-\U00002BEF\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251\U0001f926-\U0001f937\U00010000-\U0010ffff"
        u"\u2640-\u2642\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030]+",
        '', text, flags=re.UNICODE
    )
    return text.strip()

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def get_audio_stream(sr: int, sw: int):
    """Cria ou reutiliza um stream de áudio do PyAudio."""
    global active_stream, stream_sr, pa_ctx
    
    # Determinar formato PyAudio baseado no sample width (sw)
    # sw=2 -> int16, sw=4 -> float32
    pa_format = pyaudio.paInt16 if sw == 2 else pyaudio.paFloat32
    
    if pa_ctx is None:
        pa_ctx = pyaudio.PyAudio()
        
    if active_stream is not None and stream_sr == sr:
        return active_stream
        
    if active_stream is not None:
        active_stream.stop_stream()
        active_stream.close()
        
    log(f"Configurando saída: {sr}Hz, SW={sw}")
    active_stream = pa_ctx.open(
        format=pa_format,
        channels=1,
        rate=sr,
        output=True
    )
    stream_sr = sr
    return active_stream

def play_audio_chunk(audio_data: bytes, sr: int, sw: int):
    """Toca bytes brutos de áudio mantendo o stream aberto."""
    try:
        stream = get_audio_stream(sr, sw)
        stream.write(audio_data)
    except Exception as e:
        log(f"[ERRO] Falha ao tocar áudio: {e}")
        global active_stream
        active_stream = None

def record_ptt() -> bytes | None:
    """Grava áudio enquanto a tecla PTT está pressionada."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    frames = []
    log(f"🔴 Gravando... (solte [{PTT_KEY.upper()}] para enviar)")
    
    while keyboard.is_pressed(PTT_KEY):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    pa.terminate()
    
    if len(frames) < 5:  # Muito curto, ignorar
        return None
    
    # Montar WAV em memória
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
    
    log(f"✅ Gravação finalizada ({len(frames) * CHUNK / SAMPLE_RATE:.1f}s)")
    return buf.getvalue()

def send_audio(wav_bytes: bytes) -> str | None:
    """Envia o áudio para o servidor e aciona uma resposta."""
    try:
        # 1. Enviar áudio como contexto de conversa
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        r = requests.post(f"{SERVER_URL}/api/context/conversation/audio", json={
            "user": USER_NAME,
            "audio_bytes": audio_b64,
            "timestamp": int(time.time()),
            "sr": SAMPLE_RATE,
            "sw": 2,
            "ch": CHANNELS
        }, timeout=30)
        
        if r.status_code != 200:
            log(f"[ERRO] Falha ao enviar áudio: {r.text}")
            return None
        
        # 2. Acionar resposta da Gigi
        r2 = requests.post(f"{SERVER_URL}/api/response", json={}, timeout=10)
        if r2.status_code != 200:
            log(f"[ERRO] Falha ao acionar resposta: {r2.text}")
            return None
        
        job_id = r2.json().get("response", {}).get("job_id")
        log(f"📨 Job criado: {job_id}")
        return job_id
        
    except requests.exceptions.ConnectionError:
        log("[ERRO] Não foi possível conectar ao servidor. Verifique se o start_server.bat está rodando.")
        return None
    except Exception as e:
        log(f"[ERRO] Exceção ao enviar áudio: {e}")
        return None

def send_text(text: str) -> str | None:
    """Envia texto como conversa e aciona resposta."""
    try:
        r = requests.post(f"{SERVER_URL}/api/context/conversation/text", json={
            "user": USER_NAME,
            "content": text,
            "timestamp": int(time.time())
        }, timeout=10)
        
        if r.status_code != 200:
            log(f"[ERRO] Ao enviar texto: {r.text}")
            return None
        
        r2 = requests.post(f"{SERVER_URL}/api/response", json={}, timeout=10)
        if r2.status_code != 200:
            log(f"[ERRO] Ao acionar resposta: {r2.text}")
            return None
        
        return r2.json().get("response", {}).get("job_id")
    except requests.exceptions.ConnectionError:
        log("[ERRO] Servidor offline. Inicie o start_server.bat primeiro.")
        return None

# ──────────────────────────────────────────────
# WebSocket — Recebe respostas em tempo real
# ──────────────────────────────────────────────
async def websocket_listener():
    """Conecta ao WebSocket do servidor e processa eventos."""
    global stop_ws
    log("🔌 Conectando ao servidor via WebSocket...")
    
    while not stop_ws:
        try:
            async with websockets.connect(WS_URL) as ws:
                log("✅ Conectado! Aguardando respostas da Gigi...\n")
                
                while not stop_ws:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        event = json.loads(raw)
                        
                        # Estrutura do Developer Guide: { status, message, response: { job_id, finished, result: { ... } } }
                        job_type = event.get("message")
                        resp_data = event.get("response", {})
                        
                        if job_type == "response":
                            result = resp_data.get("result", {})
                            finished = resp_data.get("finished", False)
                            
                            # Texto da Gigi chegando
                            if "content" in result:
                                content = result["content"]
                                print(content, end="", flush=True)
                                if finished:
                                    print("\n")
                            
                            # Áudio chegando
                            if "audio_bytes" in result:
                                sr = result.get("sr", 16000)
                                sw = result.get("sw", 2)
                                audio_raw = base64.b64decode(result["audio_bytes"])
                                await audio_queue.put((audio_raw, sr, sw))
                        
                        # Erro reportado no job
                        if resp_data.get("finished") and not resp_data.get("success", True):
                            reason = resp_data.get("result", {}).get("reason", "desconhecido")
                            log(f"[AVISO] Resposta falhou: {reason}")
                    
                    except asyncio.TimeoutError:
                        pass  # Keepalive silencioso
                        
        except websockets.exceptions.ConnectionClosed:
            if not stop_ws:
                log("⚠️ Conexão WebSocket perdida. Reconectando em 3s...")
                await asyncio.sleep(3)
        except Exception as e:
            if not stop_ws:
                log(f"[ERRO] WebSocket: {e}. Reconectando em 3s...")
                await asyncio.sleep(3)

async def audio_player():
    """Toca os chunks de áudio recebidos em sequência."""
    while not stop_ws:
        try:
            # Recebe a tupla (bytes, sr, sw)
            item = await asyncio.wait_for(audio_queue.get(), timeout=1)
            if item:
                audio_data, sr, sw = item
                # Rodar em thread separada para não bloquear o loop de mensagens
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, play_audio_chunk, audio_data, sr, sw)
        except asyncio.TimeoutError:
            pass

# ──────────────────────────────────────────────
# Loop Principal
# ──────────────────────────────────────────────
async def main():
    global stop_ws
    
    print("=" * 50)
    print("  🎙️  Cliente de Voz - Gigi a Sapeca")
    print("=" * 50)
    print(f"  • Segure [{PTT_KEY.upper()}] para falar (Push-To-Talk)")
    print(f"  • Digite texto e pressione Enter para enviar")
    print(f"  • Digite 'sair' para encerrar")
    print("=" * 50)
    print()
    
    # Iniciar WebSocket e player em paralelo
    ws_task    = asyncio.create_task(websocket_listener())
    play_task  = asyncio.create_task(audio_player())
    
    loop = asyncio.get_event_loop()
    
    try:
        while True:
            # Input não-bloqueante
            user_input = await loop.run_in_executor(None, input, "Você: ")
            user_input = user_input.strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["sair", "exit", "quit"]:
                log("Encerrando...")
                break
            
            # Verifica se é a tecla PTT (atalho de texto para gravar)
            if user_input.lower() == "ptt":
                log(f"Pressione e segure [{PTT_KEY.upper()}] para gravar...")
                while not keyboard.is_pressed(PTT_KEY):
                    await asyncio.sleep(0.05)
                wav = await loop.run_in_executor(None, record_ptt)
                if wav:
                    send_audio(wav)
            else:
                # Texto simples
                job = await loop.run_in_executor(None, send_text, user_input)
                if not job:
                    log("[AVISO] Não foi possível enviar a mensagem.")
    
    finally:
        stop_ws = True
        ws_task.cancel()
        play_task.cancel()
        try:
            await ws_task
            await play_task
        except asyncio.CancelledError:
            pass
        log("Cliente encerrado.")

if __name__ == "__main__":
    asyncio.run(main())
