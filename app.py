import time
import hashlib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from collections import OrderedDict
from sentence_transformers import SentenceTransformer

app = FastAPI()

# =====================
# CONFIG
# =====================

CACHE_SIZE = 1500
TTL = 86400
MODEL_COST = 1.20 / 1_000_000
TOKENS_PER_REQ = 2000

# =====================
# MODELS
# =====================

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================
# CACHE STRUCTURES
# =====================

cache = OrderedDict()
embeddings = {}

# analytics
total_requests = 0
hits = 0
misses = 0
cached_tokens = 0

# =====================
# REQUEST MODEL
# =====================

class Query(BaseModel):
    query: str
    application: str


# =====================
# UTILS
# =====================

def hash_query(q):
    return hashlib.md5(q.encode()).hexdigest()

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


def clean_expired():
    now = time.time()
    expired = [k for k,v in cache.items() if now - v["time"] > TTL]
    for k in expired:
        cache.pop(k)
        embeddings.pop(k, None)


# =====================
# MAIN ENDPOINT
# =====================

@app.post("/")
def ask(q: Query):
    global total_requests,hits,misses,cached_tokens

    start = time.time()
    total_requests += 1
    clean_expired()

    key = hash_query(q.query)

    # ---------- EXACT MATCH ----------
    if key in cache:
        hits += 1
        cached_tokens += TOKENS_PER_REQ
        cache.move_to_end(key)

        return {
            "answer": cache[key]["answer"],
            "cached": True,
            "latency": int((time.time()-start)*1000),
            "cacheKey": key
        }

    # ---------- SEMANTIC CACHE ----------
    emb = embed_model.encode(q.query)

    for k,e in embeddings.items():
        if cosine(emb,e) > 0.95:
            hits += 1
            cached_tokens += TOKENS_PER_REQ
            cache.move_to_end(k)

            return {
                "answer": cache[k]["answer"],
                "cached": True,
                "latency": int((time.time()-start)*1000),
                "cacheKey": k
            }

    # ---------- MISS ----------
    misses += 1

    # simulate LLM call
    time.sleep(0.3)
    answer = f"AI response for: {q.query}"

    # store
    if len(cache) >= CACHE_SIZE:
        oldest = next(iter(cache))
        cache.pop(oldest)
        embeddings.pop(oldest,None)

    cache[key] = {"answer":answer,"time":time.time()}
    embeddings[key] = emb

    return {
        "answer": answer,
        "cached": False,
        "latency": int((time.time()-start)*1000),
        "cacheKey": key
    }


# =====================
# ANALYTICS
# =====================

@app.get("/analytics")
def analytics():
    if total_requests == 0:
        rate = 0
    else:
        rate = hits/total_requests

    savings = cached_tokens * MODEL_COST
    percent = rate*100

    return {
        "hitRate": round(rate,2),
        "totalRequests": total_requests,
        "cacheHits": hits,
        "cacheMisses": misses,
        "cacheSize": len(cache),
        "costSavings": round(savings,2),
        "savingsPercent": round(percent,2),
        "strategies":[
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    }
