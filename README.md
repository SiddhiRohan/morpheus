# Morpheus

> A multi-agent system where a local LLM fine-tunes itself in real time during a conversation.

## What Is Morpheus?

Every AI system you have ever used is frozen. Its weights never change while you talk to it. RAG gives it notes to read. Memory gives it a diary. But the underlying model — the parameters that determine how it reasons — never change during your session.

**Morpheus changes that.**

Morpheus is a multi-agent pipeline where a local LLM updates its own LoRA adapter weights in real time, turn by turn, during a live conversation. Claude API acts as an automated critic that evaluates every response, generates a better answer, and feeds it to a background training loop running concurrently on the same GPU.

The model at the end of a Morpheus conversation is not the same model that started it.

---

## Core Innovation

- **Concurrent inference and training** on a single consumer GPU
- **Claude as the critic** — generates training signal automatically, no human labeling
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
Critic Agent (Claude Sonnet 4.6) scores response 0-10
     ↓
Curator filters — queues pairs scoring below 7
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
| Local model | Qwen 2.5-3B-Instruct (4-bit quantized) |
| Fine-tuning | Unsloth + LoRA (r=16, 0.96% trainable params) |
| Critic agent | Claude API (claude-sonnet-4-6) |
| Training | HuggingFace Trainer (micro-batch, 2 steps per pair) |
| Backend | FastAPI — Phase 4 |
| Frontend | Live dashboard with Chart.js loss curve — Phase 4 |

---

## Hardware Requirements

- GPU with 6GB+ VRAM (tested on RTX 3060 Laptop)
- CUDA 11.8+
- Python 3.11
- Windows or Linux

---

## Project Structure
```
morpheus/
├── config.py              # All settings in one place
├── main.py                # Entry point — orchestrates everything
├── agents/
│   ├── critic.py          # Claude API critic agent
│   └── curator.py         # Training pair queue with blacklist
├── training/
│   └── trainer.py         # Async LoRA training loop
├── api/                   # FastAPI backend (Phase 4)
├── ui/                    # Live dashboard (Phase 4)
└── logs/                  # Training and curator logs
```

---

## Setup
```bash
# Clone and enter
git clone <repo-url>
cd morpheus

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
pip install "unsloth[cu118-torch260] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers peft trl accelerate anthropic fastapi uvicorn python-dotenv numpy datasets

# Add your API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

---

## Running Morpheus
```bash
python main.py
```

Commands during conversation:
- `stats` — show loss history, adapter version, training count
- `compare <question>` — show before/after answer comparison
- `quit` — exit

---

## How It Works

### The Critic Agent
Every response the local model generates is evaluated by Claude Sonnet. Claude returns a score 0-10, what was wrong, and the ideal answer in structured JSON. Responses scoring below 7 are queued for training.

### The Curator
Maintains a queue of (prompt, ideal_answer) training pairs. Each unique question is trained on exactly once — the blacklist prevents the model from memorizing a single pair and collapsing.

### The Training Loop
Runs in a background thread. When a pair is available, it acquires a lock, runs 2 gradient steps of LoRA fine-tuning, saves the updated adapter, and releases the lock. The entire update takes under 2 seconds.

### The Demo
Ask the same question at turn 1 and turn 8. Watch the loss curve drop between turns. The answer improves. The adapter version ticks up. The model learned during the conversation.

---

## Phase Status

- [x] Phase 1 — Environment and model setup
- [x] Phase 2 — Claude critic agent
- [x] Phase 3 — Async LoRA training loop
- [ ] Phase 4 — Live dashboard UI
- [ ] Phase 5 — Polish and demo prep
- [ ] Phase 6 — Buffer



---

## Built at HackPSU 2026