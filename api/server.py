import os
os.environ["TORCHDYNAMO_DISABLE"]            = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"]        = "1"
os.environ["UNSLOTH_DISABLE_CUSTOM_KERNELS"] = "1"

import sys
import json
import asyncio
import threading
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as morpheus

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.on_event("startup")
async def startup():
    morpheus.load_model()
    t = threading.Thread(target=morpheus.training_loop, daemon=True)
    t.start()
    print("[server] Morpheus ready.")


@app.post("/chat")
async def chat(req: ChatRequest):
    """Run inference and return response + trigger async critique."""
    answer = morpheus.chat(req.message)
    return {
        "answer"          : answer,
        "turn"            : morpheus.turn_count,
        "adapter_version" : morpheus.trainer.get_adapter_version(),
    }


@app.get("/metrics")
async def metrics():
    """Return live training metrics for dashboard."""
    loss_history    = morpheus.trainer.get_loss_history()
    adapter_version = morpheus.trainer.get_adapter_version()
    curator_stats   = morpheus.curator.stats()

    return {
        "loss_history"      : [round(l, 4) for l in loss_history],
        "adapter_version"   : adapter_version,
        "turn_count"        : morpheus.turn_count,
        "total_trained"     : curator_stats["total_queued"],
        "queue_size"        : curator_stats["current_queue"],
        "trained_questions" : curator_stats.get("trained_questions", 0),
        "timestamp"         : datetime.now().isoformat(),
    }


@app.get("/comparison")
async def comparison(question: str):
    """Return before/after answer for a question."""
    if question in morpheus.answer_log:
        log = morpheus.answer_log[question]
        return {
            "question"    : question,
            "first_answer": log.get("first", ""),
            "first_turn"  : log.get("first_turn", 0),
            "latest_answer": log.get("latest", ""),
            "latest_turn" : log.get("latest_turn", 0),
        }
    return {"question": question, "first_answer": "", "latest_answer": ""}


@app.get("/")
async def root():
    with open(os.path.join(os.path.dirname(__file__), "../ui/index.html")) as f:
        return HTMLResponse(f.read())