from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import os

app = FastAPI()

class Request(BaseModel):
    query_id: str
    query: str
    candidates: list[str]

def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/")
async def rank(req: Request):
    client = get_client()   # <-- IMPORTANT FIX

    texts = [req.query] + req.candidates

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [d.embedding for d in response.data]

    query_emb = embeddings[0]
    candidate_embs = embeddings[1:]

    scores = [
        cosine_similarity(query_emb, emb)
        for emb in candidate_embs
    ]

    ranking = np.argsort(scores)[-3:][::-1].tolist()

    return {"ranking": ranking}
