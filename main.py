from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

from prompt import SYSTEM_PROMPT
from knowledge import KNOWLEDGE_BASE

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Prestige Automations AI Assistant")

# 🔥 CORS (OBLIGATORIO PARA WORDPRESS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego se puede limitar a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    message: str

@app.post("/chat")
async def chat(q: Question):
    if not q.message or not q.message.strip():
        raise HTTPException(status_code=400, detail="Mensaje vacío")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": f"{SYSTEM_PROMPT}\n\n{KNOWLEDGE_BASE}"
                },
                {
                    "role": "user",
                    "content": q.message
                }
            ],
            max_output_tokens=400,
            temperature=0.4
        )

        # ✅ DEVOLVER RESPUESTA AL FRONTEND
        return {
            "response": response.output_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


