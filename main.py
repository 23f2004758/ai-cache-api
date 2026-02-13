from fastapi import FastAPI
from pydantic import BaseModel
import requests
import sqlite3
import datetime
import openai
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PipelineRequest(BaseModel):
    email: str
    source: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/pipeline")
def run_pipeline(request: PipelineRequest):
    return {
        "items": [],
        "notificationSent": True,
        "processedAt": datetime.datetime.utcnow().isoformat(),
        "errors": []
    }


# -------- CONFIG --------
openai.api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDQ3NThAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.z7lU4E6qvsdq5iwuu6VZHL6o4bpqv4wuAYam723cQ2Q"

DB_NAME = "pipeline.db"

# -------- DATABASE SETUP --------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_content TEXT,
            analysis TEXT,
            sentiment TEXT,
            source TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------- REQUEST MODEL --------
class PipelineRequest(BaseModel):
    email: str
    source: str


# -------- LLM FUNCTION --------
def analyze_with_llm(text):
    try:
        prompt = f"""
        Analyze this text in 2-3 sentences and classify sentiment
        as optimistic, pessimistic, or balanced.

        Text:
        {text}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content

        # Simple sentiment detection
        sentiment = "balanced"
        if "optimistic" in content.lower():
            sentiment = "optimistic"
        elif "pessimistic" in content.lower():
            sentiment = "pessimistic"

        return content, sentiment

    except Exception as e:
        return f"AI error: {str(e)}", "unknown"


# -------- STORAGE FUNCTION --------
def store_result(raw, analysis, sentiment, source):
    try:
        timestamp = datetime.datetime.utcnow().isoformat()
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO results (raw_content, analysis, sentiment, source, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (raw, analysis, sentiment, source, timestamp))
        conn.commit()
        conn.close()
        return True, timestamp
    except Exception as e:
        return False, str(e)


# -------- PIPELINE ENDPOINT --------
@app.post("/pipeline")
def run_pipeline(request: PipelineRequest):
    items_output = []
    errors = []

    try:
        response = requests.get(
            "https://jsonplaceholder.typicode.com/comments?postId=1",
            timeout=5
        )
        response.raise_for_status()
        comments = response.json()[:3]

    except Exception as e:
        return {
            "items": [],
            "notificationSent": False,
            "processedAt": datetime.datetime.utcnow().isoformat(),
            "errors": [f"API Error: {str(e)}"]
        }

    for comment in comments:
        try:
            original = comment["body"]

            analysis, sentiment = analyze_with_llm(original)

            stored, timestamp = store_result(
                original,
                analysis,
                sentiment,
                request.source
            )

            items_output.append({
                "original": original,
                "analysis": analysis,
                "sentiment": sentiment,
                "stored": stored,
                "timestamp": timestamp
            })

        except Exception as e:
            errors.append(str(e))

    # -------- Notification (Mock) --------
    try:
        print(f"Notification sent to: 23f2004758@ds.study.iitm.ac.in")
        notification_sent = True
    except Exception as e:
        errors.append(str(e))
        notification_sent = False

    return {
        "items": items_output,
        "notificationSent": notification_sent,
        "processedAt": datetime.datetime.utcnow().isoformat(),
        "errors": errors
    }
