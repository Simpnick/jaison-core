import gradio as gr
import asyncio
import os
import torch
import numpy as np
import soundfile as sf
from src.utils.operations.tts.edge import EdgeTTS
from src.utils.operations.filter_audio.rvc import RVCFilter
from src.utils.config import Config
from pydub import AudioSegment
from io import BytesIO

# Configuração de caminhos e modelos
MODELS_DIR = "models/rvc"

def get_rvc_models():
    if not os.path.exists(MODELS_DIR):
        return []
    return [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]

def get_rvc_indices():
    if not os.path.exists(MODELS_DIR):
        return []
    return [""] + [f for f in os.listdir(MODELS_DIR) if f.endswith(".index")]

async def run_rvc(input_audio, model_name, index_file_drop, index_file_path, f0_up_key, f0_method, index_rate, protect, rms_mix_rate, filter_radius, resample_sr):
    if input_audio is None:
        return None, "Por favor, forneça um áudio de entrada."
    
    sr_in, audio_in = input_audio
    
    # Priorizar o caminho manual do index se fornecido
    final_index = index_file_path if index_file_path and os.path.exists(index_file_path) else None
    if not final_index and index_file_drop:
        final_index = os.path.join(MODELS_DIR, index_file_drop)

    # Conversão para mono e int16
    if len(audio_in.shape) > 1:
        audio_in = np.mean(audio_in, axis=1)
    if audio_in.dtype != np.int16:
        audio_in = (audio_in * 32767).astype(np.int16)

    rvc = RVCFilter()
    await rvc.configure({
        "voice": model_name,
        "index_file": final_index,
        "f0_up_key": f0_up_key,
        "f0_method": f0_method,
        "index_rate": index_rate,
        "protect": protect,
        "rms_mix_rate": rms_mix_rate,
        "filter_radius": filter_radius,
        "resample_sr": resample_sr
    })
    await rvc.start()
    
    os.makedirs(os.path.dirname(Config().ffmpeg_working_src), exist_ok=True)
    temp_wav = AudioSegment(data=audio_in.tobytes(), sample_width=2, frame_rate=sr_in, channels=1)
    temp_wav.export(Config().ffmpeg_working_src, format="wav")
    
    all_audio = b""
    final_sr = 16000
    async for chunk in rvc._generate(audio_bytes=audio_in.tobytes(), sr=sr_in, sw=2, ch=1):
        all_audio += chunk["audio_bytes"]
        final_sr = chunk["sr"]
        
    if not all_audio: return None, "Falha na geração."
    return (final_sr, np.frombuffer(all_audio, dtype=np.int16)), "Processamento concluído!"

async def run_tts_rvc(text, tts_voice, model_name, index_file_drop, index_file_path, f0_up_key, f0_method, index_rate, protect, rms_mix_rate, filter_radius, resample_sr):
    if not text: return None, "Digite um texto."
    
    final_index = index_file_path if index_file_path and os.path.exists(index_file_path) else None
    if not final_index and index_file_drop:
        final_index = os.path.join(MODELS_DIR, index_file_drop)

    tts = EdgeTTS()
    await tts.configure({"voice": tts_voice})
    await tts.start()
    
    rvc = RVCFilter()
    await rvc.configure({
        "voice": model_name, "index_file": final_index, "f0_up_key": f0_up_key, "f0_method": f0_method,
        "index_rate": index_rate, "protect": protect, "rms_mix_rate": rms_mix_rate, "filter_radius": filter_radius, "resample_sr": resample_sr
    })
    await rvc.start()
    
    all_final_audio, final_sr = b"", 16000
    async for tts_chunk in tts._generate(content=text):
        async for rvc_chunk in rvc._generate(audio_bytes=tts_chunk["audio_bytes"], sr=tts_chunk["sr"], sw=tts_chunk["sw"], ch=tts_chunk["ch"]):
            all_final_audio += rvc_chunk["audio_bytes"]
            final_sr = rvc_chunk["sr"]
            
    return (final_sr, np.frombuffer(all_final_audio, dtype=np.int16)), "Texto convertido!"

def create_ui():
    models = get_rvc_models()
    indices = get_rvc_indices()
    
    with gr.Blocks(title="Gigi Lab - RVC Configurator") as demo:
        gr.Markdown("# 🎙️ Gigi Lab - Advanced Configurator")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🛠️ Configurações Principais")
                f0_up_key = gr.Slider(label="Mude o tom aqui. Se a voz for do mesmo sexo, não é necessário alterar (12 caso seja Masculino para feminino, -12 caso seja ao contrário).", minimum=-24, maximum=24, step=1, value=0)
                
                model_dropdown = gr.Dropdown(label="Selecione o modelo (.pth)", choices=models, value=models[0] if models else None)
                
                index_file_path = gr.Textbox(label="Caminho para o arquivo de Index. Deixe em branco para usar o resultado selecionado no menu debaixo:", placeholder="C:\\Caminho\\Para\\seu.index")
                index_dropdown = gr.Dropdown(label="Detecte automaticamente o caminho do Index e selecione no menu suspenso:", choices=indices, value=indices[0] if indices else "")
                
                f0_method = gr.Radio(label="Selecione o algoritmo de extração de tom 'pm': extração mais rápida; 'harvest': graves melhores; 'crepe': melhor qualidade (GPU); 'rmvpe': modelo robusto.", 
                                     choices=["pm", "harvest", "crepe", "rmvpe"], value="rmvpe")

            with gr.Column():
                gr.Markdown("### ⚙️ Ajustes Finos")
                resample_sr = gr.Slider(label="Reamostragem pós-processamento para a taxa de amostragem final, 0 significa sem reamostragem:", minimum=0, maximum=48000, step=1, value=0)
                
                rms_mix_rate = gr.Slider(label="O envelope de volume da fonte de entrada substitui a taxa de fusão do envelope de volume de saída, quanto mais próximo de 1, mais o envelope de saída é usado:", minimum=0, maximum=1, step=0.01, value=0.25)
                
                protect = gr.Slider(label="Proteja consoantes sem voz e sons respiratórios, evite artefatos como quebra de som eletrônico. Diminua para aumentar a proteção:", minimum=0, maximum=0.5, step=0.01, value=0.33)
                
                filter_radius = gr.Slider(label=">=3, use o filtro mediano para o resultado do reconhecimento do tom da heverst, o valor é o raio do filtro:", minimum=0, maximum=7, step=1, value=3)
                
                index_rate = gr.Slider(label="Taxa de recurso de recuperação (Index Rate):", minimum=0, maximum=1, step=0.01, value=0.75)

        with gr.Row():
            with gr.Column():
                with gr.Tab("🎙️ Teste com Áudio"):
                    audio_input = gr.Audio(label="Gravar ou Enviar Áudio", type="numpy")
                    btn_audio = gr.Button("Transformar voz", variant="primary")
                    audio_output = gr.Audio(label="Resultado RVC")
                    
                with gr.Tab("⌨️ Teste com Texto"):
                    tts_voice = gr.Dropdown(label="Voz do Edge (Base)", choices=["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"], value="pt-BR-FranciscaNeural")
                    text_input = gr.Textbox(label="Texto para falar", placeholder="Oi! Eu sou a Gigi...")
                    btn_text = gr.Button("Gerar e Transformar", variant="primary")
                    text_output = gr.Audio(label="Resultado RVC")
                
                status_msg = gr.Label(value="Pronto")

        # Eventos
        common_inputs = [model_dropdown, index_dropdown, index_file_path, f0_up_key, f0_method, index_rate, protect, rms_mix_rate, filter_radius, resample_sr]
        btn_audio.click(fn=run_rvc, inputs=[audio_input] + common_inputs, outputs=[audio_output, status_msg])
        btn_text.click(fn=run_tts_rvc, inputs=[text_input, tts_voice] + common_inputs, outputs=[text_output, status_msg])

    return demo

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    gui = create_ui()
    gui.launch(inbrowser=True, share=False)
