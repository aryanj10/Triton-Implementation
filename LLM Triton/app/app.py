from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

openai.api_base = os.getenv("OPENAI_BASE", "http://localhost:8008/v1")
openai.api_key = "sk-no-auth"

app = FastAPI()
chat_history = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    if req.user_id not in chat_history:
        chat_history[req.user_id] = []
    messages = chat_history[req.user_id] + [{"role": "user", "content": req.message}]
    response = openai.ChatCompletion.create(
        model="models/vllm_model/1",
        messages=messages,
        temperature=0.7
    )
    reply = response["choices"][0]["message"]["content"]
    chat_history[req.user_id].append({"role": "user", "content": req.message})
    chat_history[req.user_id].append({"role": "assistant", "content": reply})
    return {"response": reply, "history": chat_history[req.user_id]}

class EmbeddingRequest(BaseModel):
    inputs: list

@app.post("/embed")
async def embed(req: EmbeddingRequest):
    response = openai.Embedding.create(
        model="models/vllm_model/1",
        input=req.inputs
    )
    return response
