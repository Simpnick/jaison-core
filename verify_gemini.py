import asyncio
import os
from dotenv import load_dotenv
import sys
import datetime

# Add src to path
sys.path.append(os.path.abspath("src"))

from utils.operations.t2t.gemini import GeminiT2T
from utils.operations.embedding.gemini import GeminiEmbedding
from utils.prompter.message import ChatMessage

async def test_gemini():
    load_dotenv()
    
    print("--- Testing Gemini T2T ---")
    gemini_t2t = GeminiT2T()
    await gemini_t2t.start()
    await gemini_t2t.configure({"model": "models/gemma-3-1b-it"})
    
    instruction = "Você é um assistente prestativo."
    messages = [
        ChatMessage(user="User", message="Olá, responda curto: qual a cor do céu?", time=datetime.datetime.now())
    ]
    
    try:
        async for chunk in gemini_t2t._generate(instruction_prompt=instruction, messages=messages):
            print(chunk["content"], end="", flush=True)
        print("\nT2T Success!")
    except Exception as e:
        print(f"\nT2T Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await gemini_t2t.close()

    print("\n--- Testing Gemini Embedding ---")
    gemini_emb = GeminiEmbedding()
    await gemini_emb.start()
    await gemini_emb.configure({"model": "models/gemini-embedding-2-preview"})
    
    try:
        async for chunk in gemini_emb._generate(content="Hello world"):
            emb_b64 = chunk["embedding"]
            print(f"Embedding generated (length: {len(emb_b64)})")
            print(f"Prefix: {emb_b64[:20]}...")
        print("Embedding Success!")
    except Exception as e:
        print(f"Embedding Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await gemini_emb.close()

if __name__ == "__main__":
    asyncio.run(test_gemini())
