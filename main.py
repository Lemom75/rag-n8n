from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

app = FastAPI()

# Load credentials
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Request model
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

@app.post("/ask")
def ask_qna(request: AskRequest):
    try:
        # 1. Embedding
        embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=request.question
        )
        vector = embed.data[0].embedding

        # 2. Pinecone query
        result = index.query(
            vector=vector,
            top_k=request.top_k,
            include_metadata=True
        )
        contexts = [m.metadata["text"] for m in result.matches]
        context_str = "\n\n".join(contexts)

        # 3. Completion with context
        system_prompt = "Tu es un assistant spécialisé n8n. Ne réponds que si la réponse est dans le contexte."
        user_prompt = f"Contexte:\n{context_str}\n\nQuestion: {request.question}"

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        try:
    return {"answer": completion.choices[0].message.content}
except Exception as err:
    print("Réponse OpenAI brute :", completion)
    return {"answer": f"Erreur côté API (structure inattendue): {str(err)}"}


    except Exception as e:
        return {"answer": f"Erreur côté API : {str(e)}"}
