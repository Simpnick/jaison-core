import asyncio
import requests
import json
import websockets
import threading
import time
import sys

# Configurações do Servidor (ajuste se necessário)
HOST = "127.0.0.1"
PORT = 7272
SERVER_URL = f"http://{HOST}:{PORT}"
WS_URL = f"ws://{HOST}:{PORT}"

async def listen_ws():
    """Escuta o WebSocket para receber as respostas do JAIson em tempo real."""
    backoff = 1
    while True:
        try:
            async with websockets.connect(WS_URL) as websocket:
                print(f"\n[WebSocket: Conectado ao servidor]")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # O servidor envia eventos com id='response' para T2T
                    if data.get('id') == 'response':
                        result = data.get('result', {})
                        
                        # 'content' contém os pedaços da resposta filtrada (final)
                        if 'content' in result:
                            print(f"{result['content']}", end="", flush=True)
                        elif 'content' in data: # Support flattened structure too
                            print(f"{data['content']}", end="", flush=True)
                        
                        # Quando o job termina
                        if data.get('finished'):
                            if data.get('success'):
                                pass # Silently finish to keep it clean
                            else:
                                reason = data.get('reason') or (result.get('reason') if isinstance(result, dict) else None)
                                print(f"\n[Erro na Gigi: {reason}]")
                            print("\nVocê: ", end="", flush=True)
                            
        except Exception as e:
            print(f"\n[WebSocket: Desconectado ({e}). Tentando reconectar em {backoff}s...]")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


def start_ws_thread():
    """Inicia o loop de eventos do WebSocket em uma thread separada."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(listen_ws())

def main():
    print(f"--- Cliente de Chat (Gigi a Sapeca) ---")
    print(f"Conectando ao servidor em {SERVER_URL}")
    print("Digite 'sair' para encerrar ou 'limpar' para resetar o contexto.")
    
    # Inicia a thread do WebSocket
    thread = threading.Thread(target=start_ws_thread, daemon=True)
    thread.start()
    
    # Pequena pausa para o WS conectar
    time.sleep(1)
    
    try:
        # Initial prompt
        print("\nVocê: ", end="", flush=True)
        while True:
            user_input = input()
            
            if not user_input.strip():
                print("Você: ", end="", flush=True)
                continue
                
            if user_input.lower() == 'sair':
                break
            
            if user_input.lower() == 'limpar':
                requests.delete(f"{SERVER_URL}/api/context")
                print("[Contexto limpo]")
                print("\nVocê: ", end="", flush=True)
                continue
            
            # 1. Adiciona a mensagem do usuário ao histórico (Contexto)
            try:
                resp_ctx = requests.post(f"{SERVER_URL}/api/context/conversation/text", json={
                    "user": "User",
                    "content": user_input,
                    "timestamp": int(time.time())
                })
                if resp_ctx.status_code != 200:
                    print(f"[Erro ao adicionar contexto: {resp_ctx.text}]")
                    print("\nVocê: ", end="", flush=True)
                    continue
                
                # 2. Solicita uma resposta da Gigi
                print("Gigi: ", end="", flush=True)
                resp_job = requests.post(f"{SERVER_URL}/api/response", json={})
                if resp_job.status_code != 200:
                    print(f"\n[Erro ao solicitar resposta: {resp_job.text}]")
                    print("\nVocê: ", end="", flush=True)
            
            except Exception as e:
                print(f"\n[Erro de conexão: {e}]")
                print("\nVocê: ", end="", flush=True)
                
    except KeyboardInterrupt:
        pass
    print("\nEncerrando cliente.")

if __name__ == "__main__":
    main()
