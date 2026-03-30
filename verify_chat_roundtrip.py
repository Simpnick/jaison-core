import asyncio
import json
import websockets
import requests
import time
import sys

# Configurações
SERVER_URL = "http://127.0.0.1:7272"
WS_URL = "ws://127.0.0.1:7272"

async def test_roundtrip():
    print("--- Iniciando Teste de Roundtrip JAIson ---")
    
    # 1. Limpar contexto para teste limpo
    print("Limpando contexto...")
    requests.delete(f"{SERVER_URL}/api/context")
    
    # 2. Conectar ao WebSocket
    print(f"Conectando ao WebSocket em {WS_URL}...")
    async with websockets.connect(WS_URL) as websocket:
        print("Conectado!")
        
        # 3. Enviar mensagem via HTTP
        print("Enviando mensagem: 'Olá JAIson, você está funcionando?'")
        resp = requests.post(f"{SERVER_URL}/api/context/conversation/text", json={
            "user": "Tester",
            "content": "Olá JAIson, você está funcionando?",
            "timestamp": int(time.time())
        })
        
        if resp.status_code != 200:
            print(f"Erro ao enviar mensagem: {resp.text}")
            return
            
        # 4. Solicitar resposta
        requests.post(f"{SERVER_URL}/api/response", json={})
        
        # 5. Escutar WebSocket
        full_response = ""
        success = False
        print("Aguardando resposta do JAIson...")
        
        try:
            # Timeout de 30 segundos
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                data = json.loads(message)
                
                # Debug payload
                # print(f"Evento recebido: {data.get('id')}")
                
                if data.get('id') == 'response':
                    result = data.get('result', {})
                    if 'content' in result:
                        content = result['content']
                        full_response += content
                        print(content, end="", flush=True)
                    
                    if data.get('finished'):
                        print("\n[Job Finalizado]")
                        success = data.get('success', False)
                        break
        except asyncio.TimeoutError:
            print("\n[Erro] Timeout aguardando resposta do WebSocket.")
        
        print("\n--- Resultado do Teste ---")
        if success and len(full_response) > 0:
            print(f"SUCESSO! Resposta recebida: {full_response[:100]}...")
            sys.exit(0)
        else:
            print(f"FALHA! Sucesso: {success}, Tamanho da resposta: {len(full_response)}")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_roundtrip())
