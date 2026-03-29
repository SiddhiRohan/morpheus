# Morpheus

> A local LLM that fine-tunes itself in real time during a conversation.

## Overview

Every AI system you have ever used is frozen. Its weights never change while you talk to it. RAG gives it notes to read. Memory gives it a diary. But the underlying model — the parameters that determine how it reasons — never change during your session.

**Morpheus changes that.**

Morpheus is a multi-agent pipeline where a local LLM updates its own LoRA adapter weights in real time, turn by turn, during a live conversation. Claude API acts as an automated critic that evaluates every response, generates a better answer, and feeds it to a background training loop running concurrently on the same GPU.

The model at the end of a Morpheus conversation is not the same model that started it.

---

## Core Innovation

- **Concurrent inference and training** on a single consumer GPU
- **Claude as the critic** — generates training signal automatically, no human labeling required
- **LoRA hot-swap** — adapter weights update between turns without reloading the model
- **Live loss curve** — training is visible and measurable in real time
- **Curator blacklist** — each question trained on exactly once, prevents overfitting collapse

---

## Architecture
```
User message
     ↓
Orchestrator (main.py)
     ↓
Local LLM generates response (Qwen 2.5-3B, 4-bit)
     ↓
Critic Agent (Claude Sonnet) scores response 0-10
     ↓
Curator filters — queues pairs scoring below threshold
     ↓
LoRA Trainer runs async on background thread
     ↓
Adapter weights updated — model is now different
     ↓
Next turn uses updated model
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Local model | Qwen 2.5-3B-Instruct (4-bit quantized via Unsloth) |
| Fine-tuning | LoRA — r=16, targeting all attention and MLP layers |
| Critic agent | Claude API (Sonnet) |
| Training | HuggingFace Trainer, micro-batch, async background thread |
| Backend | FastAPI |
| Frontend | Vanilla JS + Chart.js live dashboard |

---

## Requirements

- NVIDIA GPU with 6GB+ VRAM
- CUDA 11.8+
- Python 3.11

---

## Project Structure
```
morpheus/
├── config.py              # All settings
├── main.py                # Orchestrator and inference loop
├── agents/
│   ├── critic.py          # Claude API critic agent
│   └── curator.py         # Training pair queue
├── training/
│   └── trainer.py         # Async LoRA training loop
├── api/
│   └── server.py          # FastAPI backend
└── ui/
    └── index.html         # Live training dashboard
```

---

## Setup
```bash
git clone https://github.com/SiddhiRohan/morpheus.git
cd morpheus

python -m venv venv
source venv/Scripts/activate

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
pip install "unsloth[cu118-torch260] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers peft trl accelerate anthropic fastapi uvicorn python-dotenv numpy datasets

echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

---

## Running
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

---

## How It Works

### Critic Agent
Every response is evaluated by Claude. It returns a score 0-10, what was wrong, and the ideal answer in structured JSON. Responses scoring below the threshold are queued for training.

### Curator
Maintains a queue of training pairs. Each unique question is trained on exactly once — preventing the model from collapsing by memorizing a single example.

### Training Loop
Runs in a background thread. Acquires a lock, runs gradient steps of LoRA fine-tuning, saves the updated adapter, releases the lock. Each update takes under 2 seconds on a consumer GPU.

### Dashboard
A live web interface showing the conversation, loss curve updating in real time, adapter version counter, and a before/after comparison panel proving the model improved.