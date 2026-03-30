import asyncio
import os
import sys
from dotenv import load_dotenv
from src.utils.operations.manager import OperationManager
from src.utils.config import Config
from pydub import AudioSegment
from io import BytesIO
import winsound
import re

# Carregar variaveis de ambiente (.env)
load_dotenv()

def clean_text(text):
    # Remove tags XML como <emotion>, <neutral>, etc.
    text = re.sub(r'<[^>]+>', '', text)
    # Remove tags de emote como [[emote:sorriso]]
    text = re.sub(r'\[\[emote:[^\]]+\]\]', '', text)
    # Remove emojis e símbolos especiais (Unicode ranges)
    text = re.sub(
        u"[\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"   # Símbolos e pictogramas
        u"\U0001F680-\U0001F6FF"   # Transporte e mapas
        u"\U0001F1E0-\U0001F1FF"   # Bandeiras
        u"\U00002500-\U00002BEF"   # Caracteres CJK
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d\u23cf\u23e9\u231a\ufe0f\u3030]+",
        '', text, flags=re.UNICODE
    )
    # Remove pensamentos entre parênteses (opcionalmente - deixe comentado se quiser que ela leia)
    # text = re.sub(r'\(.*?\)', '', text)
    return text.strip()

async def chat_test():
    print("===========================================")
    print("   Teste de Conversa: Gigi a Sapeca")
    print("===========================================\n")
    
    # Inicializar Gerenciador de Operações e Configurações
    from src.utils.operations.manager import load_op, OpTypes
    from src.utils.config import Config
    
    # Carregar o arquivo config.yaml
    Config().load_from_name("config")
    
    from src.utils.operations.manager import load_op, OpTypes
    from utils.prompter import Prompter
    
    # Carregar Personalidade da Gigi
    print("Carregando personalidade da Gigi...")
    prompter = Prompter()
    # Configurar o prompter com os dados do config.yaml
    await prompter.configure(Config().prompter)
    
    # Combinar instrução e personalidade usando os métodos corretos
    system_prompt = f"{prompter.get_instructions_prompt()}\n\n{prompter.get_character_prompt()}"

    def get_config(role, op_id):
        for op in Config().operations:
            if op.get("role") == role and op.get("id") == op_id:
                return op
        return {}

    # Carregar T2T (Gemini)
    print("Conectando ao Gemini...")
    t2t_cfg = get_config("t2t", "gemini")
    t2t = load_op(OpTypes.T2T, "gemini")
    await t2t.configure(t2t_cfg)
    await t2t.start()
    
    # Carregar TTS (Edge)
    print("Iniciando Edge TTS...")
    tts_cfg = get_config("tts", "edge")
    tts = load_op(OpTypes.TTS, "edge")
    await tts.configure(tts_cfg)
    await tts.start()
    
    # Carregar Filtro (RVC)
    print("Ativando Garganta Virtual (RVC)...")
    rvc_cfg = get_config("filter_audio", "rvc")
    rvc = load_op(OpTypes.FILTER_AUDIO, "rvc")
    await rvc.configure(rvc_cfg)
    await rvc.start()
    
    print("\n[Gigi] Pronta! Digite sua mensagem (ou 'sair' para encerrar):\n")
    
    while True:
        user_input = input("Voce: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            break
            
        print("\n[Gigi] Pensando...", end="\r")
        
        # 1. Gerar Texto (T2T)
        full_response = ""
        try:
            async for text_chunk in t2t._generate(instruction_prompt=system_prompt, messages=[user_input]):
                full_response += text_chunk["content"]
        except Exception as e:
            if "429" in str(e):
                print("\n[ERRO] O Google bloqueou o pedido (Limite de Cota atingido). Aguarde alguns segundos e tente novamente.")
            else:
                print(f"\n[ERRO] Falha na IA: {e}")
            continue
        
        print(f"[Gigi] {full_response}\n")
        print("[Gigi] Falando...", end="\r")
        
        # 2. Gerar Voz e Filtrar
        all_final_audio = b""
        sr, sw, ch = 16000, 2, 1
        
        # Limpar tags XML antes de enviar para o TTS
        tts_input = clean_text(full_response)
        
        async for tts_chunk in tts._generate(content=tts_input):
            # Passar o audio do TTS para o Filtro RVC
            async for rvc_chunk in rvc._generate(
                audio_bytes=tts_chunk["audio_bytes"],
                sr=tts_chunk["sr"],
                sw=tts_chunk["sw"],
                ch=tts_chunk["ch"]
            ):
                all_final_audio += rvc_chunk["audio_bytes"]
                sr, sw, ch = rvc_chunk["sr"], rvc_chunk["sw"], rvc_chunk["ch"]
        
        # 3. Tocar Som
        if all_final_audio:
            audio = AudioSegment(
                data=all_final_audio,
                sample_width=sw,
                frame_rate=sr,
                channels=ch
            )
            wav_io = BytesIO()
            audio.export(wav_io, format="wav")
            winsound.PlaySound(wav_io.getvalue(), winsound.SND_MEMORY)
        
        print("                                     ", end="\r") # Limpar linha do "Falando..."

if __name__ == "__main__":
    # Ajustar diretório de trabalho
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Forçar o uso das variáveis de ambiente corretas para o RVC
    os.environ["weight_root"] = "models/rvc"
    os.environ["hubert_path"] = "models/rvc/hubert_base.pt"
    os.environ["rmvpe_root"] = "models/rvc"
    os.environ["index_root"] = "models/rvc"

    try:
        asyncio.run(chat_test())
    except KeyboardInterrupt:
        print("\nSessão encerrada.")
