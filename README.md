<div align="center">

```
███╗   ███╗ ██████╗ ██████╗ ██████╗ ██╗  ██╗███████╗██╗   ██╗███████╗
████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ██║██╔════╝██║   ██║██╔════╝
██╔████╔██║██║   ██║██████╔╝██████╔╝███████║█████╗  ██║   ██║███████╗
██║╚██╔╝██║██║   ██║██╔══██╗██╔═══╝ ██╔══██║██╔══╝  ██║   ██║╚════██║
██║ ╚═╝ ██║╚██████╔╝██║  ██║██║     ██║  ██║███████╗╚██████╔╝███████║
╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝
```

**A local LLM that fine-tunes itself in real time during a conversation.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Unsloth](https://img.shields.io/badge/Unsloth-LoRA-FF6B35?style=flat-square)](https://github.com/unslothai/unsloth)
[![Claude](https://img.shields.io/badge/Claude-Sonnet-D4A574?style=flat-square)](https://anthropic.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

<div align="center">
  <img src="assets/dashboard.jpg" alt="Morpheus Live Dashboard" width="900"/>
</div>

---

## The Problem With Every AI Today

Every AI system you have ever used is **frozen**.

ChatGPT. Claude. Gemini. Llama. The moment these models finish training, they stop learning. You can give them memory — they remember what you said. You can give them retrieval — they look things up. But the underlying model, the billions of parameters that determine *how it reasons*, never change while you talk to it.

This means every AI system in the world has a fundamental ceiling. It can retrieve. It can remember. But it cannot **learn**.

**Morpheus removes that ceiling.**

---

## What Morpheus Does

Morpheus is a multi-agent pipeline where a local LLM updates its own LoRA adapter weights **in real time**, turn by turn, during a live conversation. Claude API acts as an automated critic — evaluating every response, generating a better answer, and feeding it to a training loop running concurrently on the same GPU.

The model at the end of a Morpheus conversation is not the same model that started it.

```
Turn 1  →  Model answers  →  Score: 5/10  →  Trained  →  Loss: 2.8
Turn 2  →  Model answers  →  Score: 5/10  →  Trained  →  Loss: 2.4
Turn 3  →  Model answers  →  Score: 6/10  →  Trained  →  Loss: 2.1
Turn 4  →  Model answers  →  Score: 7/10  →  Skipped  →  (already good)
Turn 5  →  Model answers  →  Score: 8/10  →  Skipped  →  (already good)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MORPHEUS PIPELINE                           │
└─────────────────────────────────────────────────────────────────────┘

  User Input
      │
      ▼
┌─────────────┐
│ Orchestrator │  main.py — routes messages, manages state, owns locks
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│                  LOCAL MODEL LAYER                   │
│                                                     │
│   ┌───────────────────────────────────────────┐    │
│   │  Qwen 2.5-3B  (4-bit NF4 quantized)       │    │
│   │  Base weights: FROZEN                     │    │
│   │                          +                │    │
│   │  LoRA Adapters  (r=16, 0.96% of params)  │    │
│   │  Adapter weights: TRAINABLE               │    │
│   └───────────────────────────────────────────┘    │
│                     │                               │
│              Generates response                     │
└─────────────────────┼───────────────────────────────┘
                       │
          ┌────────────┴─────────────┐
          │                          │
          ▼                          ▼
   Response to User          ┌───────────────┐
                             │  CRITIC AGENT  │
                             │               │
                             │  Claude Sonnet │
                             │  via API       │
                             │               │
                             │  Returns:      │
                             │  • Score 0-10  │
                             │  • What wrong  │
                             │  • Ideal answer│
                             └───────┬───────┘
                                     │
                                     ▼
                             ┌───────────────┐
                             │    CURATOR     │
                             │               │
                             │  • Score < 8? │
                             │  • First time?│
                             │  • Queue it   │
                             └───────┬───────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │         TRAINER (async)         │
                    │                                │
                    │  Background thread             │
                    │  Acquires swap_lock            │
                    │  2 gradient steps              │
                    │  Saves adapter v(N+1)          │
                    │  Releases lock                 │
                    │  Duration: ~2 seconds          │
                    └────────────────────────────────┘
                                     │
                                     ▼
                         LoRA weights hot-swapped
                         Model is now different
                         Loss curve updates on dashboard
```

### Why This Architecture Is Novel

The hard engineering problem is not any single component — it is running **inference and training simultaneously on the same GPU** without memory collisions or inference interruption.

| Challenge | Solution |
|---|---|
| Concurrent CUDA access | `threading.Lock` around `model.generate` and `trainer.train` |
| Weight swap during inference | Lock acquired between turns, never mid-generation |
| Overfitting on one example | Curator blacklist — each question trained exactly once |
| Training signal quality | Claude generates ideal answers, not human labelers |
| Windows Triton incompatibility | Surgical patches to 4 Unsloth kernel files |

---

## Model Choices — And Why

### Local Model: Qwen 2.5-3B-Instruct (4-bit)

We evaluated Phi-4-mini, Llama 3.2-3B, and Qwen 2.5-3B before committing. The requirements were strict:

- Must fit in 6GB VRAM alongside the training loop
- Must produce coherent answers a critic can meaningfully evaluate
- Must be stable under repeated micro-batch LoRA updates

**Phi-4-mini** was our first choice. It failed — catastrophic output degradation (repetition loops, incoherent text) after 5 training steps due to quantization instability under concurrent LoRA updates.

**Qwen 2.5-3B** held up cleanly. At 4-bit NF4 quantization it uses ~2.3GB VRAM, leaving 3.7GB for the training loop. Its attention architecture is stable under LoRA updates in a way other small models are not. At 3B parameters it is large enough to give answers worth evaluating.

### Critic Model: Claude Sonnet (API)

The critic does not need to be local or trainable. It needs to be accurate. Claude Sonnet scores responses reliably on a 0-10 scale, identifies specific errors, and generates ideal answers short enough to serve as training targets.

The deliberate separation matters: the local model is **fast and trainable but limited**. Claude is **powerful and accurate but frozen**. Morpheus combines both — neither could do this alone.

### Fine-Tuning: LoRA (Low-Rank Adaptation)

LoRA adds small trainable matrices (rank 16) to the attention and MLP layers — about 0.96% of total parameters. This means:

- Base model weights never change — no catastrophic forgetting
- Adapter weights are small enough to update in 2 seconds on consumer hardware
- Adapters can be saved and swapped without reloading the full model

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Local model | Qwen 2.5-3B-Instruct (4-bit) | Stable under LoRA, fits 6GB VRAM |
| Quantization | Unsloth + bitsandbytes NF4 | 4x memory reduction, minimal accuracy loss |
| Fine-tuning | LoRA r=16 via PEFT | Fast updates, no catastrophic forgetting |
| Training | HuggingFace Trainer | Standard API, bypasses broken Triton kernels |
| Critic | Claude Sonnet via Anthropic API | Most reliable scorer for structured JSON output |
| Backend | FastAPI + uvicorn | Async, fast, simple to deploy |
| Frontend | Vanilla JS + Chart.js | Zero build step, works offline, loads instantly |

---

## Requirements

- NVIDIA GPU with **6GB+ VRAM** (tested on RTX 3060 Laptop)
- **CUDA 11.8+**
- **Python 3.11**
- An **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com)

---

## Project Structure

```
morpheus/
├── config.py              # All hyperparameters and settings
├── main.py                # Orchestrator — inference loop, threading, state
├── agents/
│   ├── critic.py          # Claude API critic — scores and generates ideal answers
│   └── curator.py         # Training queue with per-question blacklist
├── training/
│   └── trainer.py         # Async LoRA trainer — runs concurrently with inference
├── api/
│   └── server.py          # FastAPI — /chat, /metrics, /comparison endpoints
└── ui/
    └── index.html         # Live dashboard — loss curve, chat, before/after panel
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SiddhiRohan/morpheus.git
cd morpheus
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
source venv/Scripts/activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies

Run each block in order and wait for completion before the next.

```bash
pip install --upgrade pip setuptools wheel
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
# Windows only
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl

# Linux / Mac
pip install bitsandbytes
```

```bash
pip install "unsloth[cu118-torch260] @ git+https://github.com/unslothai/unsloth.git"
```

```bash
pip install transformers peft trl accelerate anthropic fastapi uvicorn python-dotenv numpy datasets
```

### 4. Add your API key

```bash
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### 5. Start Morpheus

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete`, then open:

```
http://localhost:8000
```

---

## Usage

The dashboard has three panels:

| Panel | What it shows |
|---|---|
| Left — Conversation | Chat interface with turn counter and adapter version per response |
| Center — Training Metrics | Live loss curve, adapter version, pairs trained, training log |
| Right — Before / After | Automatic comparison of turn 1 vs latest answer for each question |

**To trigger training:** Ask questions the model answers imperfectly. Scores below 8/10 get queued. The loss curve updates after each training cycle.

**To see improvement:** Ask the same question at turn 1 and turn 6+. The before/after panel shows the difference automatically.

---

## How Each Component Works

### Critic Agent (`agents/critic.py`)
Sends every (question, answer) pair to Claude Sonnet with a strict JSON schema. Returns `score`, `what_was_wrong`, and `ideal_answer`. The ideal answer is capped at 120 words — short enough to be a reachable training target for a 3B model generating 200 tokens.

### Curator (`agents/curator.py`)
Maintains a `deque` of training pairs with a per-question blacklist (`trained_questions: set`). Once a question is trained on, it is permanently skipped — preventing the weight collapse that happens when a model memorizes one example over and over.

### Trainer (`training/trainer.py`)
Uses HuggingFace `Trainer` with `batch_size=1` and `max_steps=2`. Runs in a `threading.Thread`. Acquires `swap_lock` before training so inference cannot read weights during an update. Saves each adapter version to `model/adapter/vN/`. Patches Unsloth's custom cross-entropy loss with standard PyTorch to avoid Triton kernel failures on Windows.

### Server (`api/server.py`)
Three endpoints: `POST /chat` triggers inference and returns the answer with current adapter version. `GET /metrics` returns loss history, adapter version, queue size, and turn count — polled every 2 seconds by the dashboard. `GET /comparison` returns before/after answers for any question.

---

## Contributing

Contributions are welcome.

### Reporting issues

Open an issue with:
- OS, GPU model, and VRAM
- CUDA version (`nvcc --version`)
- Exact error message and file
- Steps to reproduce

### Making changes

```bash
# Fork on GitHub, then:
git clone https://github.com/your-username/morpheus.git
cd morpheus
git checkout -b feat/your-feature-name

# Make changes, test end to end, then:
git commit -m "feat: what you changed and why"
git push origin feat/your-feature-name
# Open a pull request
```

### Areas open for contribution

- **New model support** — Llama 3.2, Mistral, Gemma architectures
- **Better critic prompts** — improving training signal quality
- **Training strategies** — GRPO, DPO, different LoRA configurations
- **Linux / Mac support** — documenting setup on non-Windows systems
- **Performance** — reducing training latency, improving inference speed
- **UI improvements** — new visualizations, mobile layout

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Built with [Unsloth](https://github.com/unslothai/unsloth) · [Anthropic Claude API](https://anthropic.com) · [HuggingFace Transformers](https://huggingface.co/transformers) · [FastAPI](https://fastapi.tiangolo.com) · [Qwen 2.5](https://huggingface.co/Qwen)

---

<div align="center">

**Made at HackPSU 2026**

</div>
