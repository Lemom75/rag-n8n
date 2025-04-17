import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# Clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pinecone_client.Index(os.getenv("PINECONE_INDEX"))

app = FastAPI()

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_qna(payload: Question):
    question = payload.query

    # 1. Embedding de la question
    embedding_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    vector = embedding_response.data[0].embedding

    # 2. Recherche dans Pinecone
    results = pinecone_index.query(vector=vector, top_k=5, include_metadata=True)
    context = "\n\n".join([match["metadata"]["text"] for match in results.matches])

    # 3. Appel OpenAI avec contexte
    prompt = f"""Tu es un assistant n8n très précis. Utilise uniquement les informations ci-dessous pour répondre.

Contexte :
{context}

Question : {question}
Réponse :"""

    completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {"answer": completion.choices[0].message.content}
