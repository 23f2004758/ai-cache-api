from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import numpy as np
import os
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 IMPORTANT — Get key from Vercel environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------- Legal Documents --------
topics = [
    "liability clause",
    "indemnification clause",
    "termination conditions",
    "confidentiality agreement",
    "governing law provision",
    "arbitration clause",
    "payment terms",
    "intellectual property rights",
    "force majeure provision",
    "non-compete agreement"
]

documents = []

for i in range(62):
    topic = topics[i % len(topics)]
    documents.append({
        "id": i,
        "content": f"""
        This contract includes a detailed {topic}.
        Breach of this {topic} may result in damages or termination.
        """
    })

doc_embeddings = None

def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def compute_embeddings():
    global doc_embeddings
    if doc_embeddings is None:
        doc_embeddings = [
            np.array(get_embedding(doc["content"]))
            for doc in documents
        ]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class SearchRequest(BaseModel):
    query: str
    k: int = 6
    rerank: bool = True
    rerankK: int = 4

@app.post("/semantic-search")
def semantic_search(request: SearchRequest):
    start = time.time()

    compute_embeddings()

    query_embedding = np.array(get_embedding(request.query))

    scores = []
    for idx, emb in enumerate(doc_embeddings):
        sim = cosine_sim(query_embedding, emb)
        scores.append((idx, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_k = scores[:request.k]

    results = [
        {
            "id": documents[idx]["id"],
            "score": round(float(score), 3),
            "content": documents[idx]["content"],
            "metadata": {"source": "legal_corpus"}
        }
        for idx, score in top_k[:request.rerankK]
    ]

    latency = int((time.time() - start) * 1000)

    return {
        "results": results,
        "reranked": False,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
