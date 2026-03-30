import pyttsx3
import sys

def test_speech():
    print("Iniciando teste do PyTTSX3...")
    try:
        engine = pyttsx3.init()
        
        # Listar vozes disponíveis no console
        voices = engine.getProperty('voices')
        print("\nVozes encontradas:")
        for idx, voice in enumerate(voices):
            print(f"[{idx}] Nome: {voice.name} | ID: {voice.id}")
        
        # Texto de teste
        text = "Olá! Este é um teste de áudio da Gigi a Sapeca usando o motor nativo do Windows."
        print(f"\nFalando: \"{text}\"")
        
        # Configurações básicas
        engine.setProperty('rate', 180)    # Velocidade (padrão é 200)
        engine.setProperty('volume', 1.0)  # Volume (0.0 a 1.0)
        
        engine.say(text)
        engine.runAndWait()
        print("\nTeste concluído com sucesso!")
        
    except Exception as e:
        print(f"\nErro ao iniciar o motor de voz: {e}")

if __name__ == "__main__":
    test_speech()
