# Morpheus

> A local LLM that fine-tunes itself in real time during a conversation.

## Overview

Every AI system you have ever used is frozen. Its weights never change while you talk to it. RAG gives it notes to read. Memory gives it a diary. But the underlying model — the parameters that determine how it reasons — never change during your session.

**Morpheus changes that.**

Morpheus is a multi-agent pipeline where a local LLM updates its own LoRA adapter weights in real time, turn by turn, during a live conversation. Claude API acts as an automated critic that evaluates every response, generates a better answer, and feeds it to a background training loop running concurrently on the same GPU.

The model at the end of a Morpheus conversation is not the same model that started it.

---

## How It Works

### Critic Agent
Every response is evaluated by Claude. It returns a score 0–10, what was wrong, and an ideal answer in structured JSON. Responses scoring below the threshold are queued for training.

### Curator
Maintains a queue of training pairs. Each unique question is trained on exactly once — preventing the model from collapsing by memorizing a single example.

### Training Loop
Runs in a background thread. Acquires a lock, runs gradient steps of LoRA fine-tuning, saves the updated adapter, releases the lock. Each update takes under 2 seconds on a consumer GPU.

### Dashboard
A live web interface showing the conversation, loss curve updating in real time, adapter version counter, and a before/after comparison panel showing the model improved.

---

## Architecture
```
User message
     ↓
Orchestrator (main.py)
     ↓
Local LLM generates response (Qwen 2.5-3B, 4-bit quantized)
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
- An Anthropic API key

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

Run each block in order and wait for each to complete before the next.
```bash
pip install --upgrade pip setuptools wheel
```
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```bash
# Windows only — skip this on Linux/Mac and use standard bitsandbytes instead
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```
```bash
pip install "unsloth[cu118-torch260] @ git+https://github.com/unslothai/unsloth.git"
```
```bash
pip install transformers peft trl accelerate anthropic fastapi uvicorn python-dotenv numpy datasets
```

### 4. Add your API key

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

You can get an API key at [console.anthropic.com](https://console.anthropic.com).

### 5. Run the server
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete` in the terminal, then open your browser at:
```
http://localhost:8000
```

---

## Usage

Once the dashboard is open:

- Type any question in the chat input and press Enter or click Send
- Watch the loss curve in the center panel update as the model trains
- The adapter version counter increments with every training cycle
- Ask the same question again after several turns — the Before/After panel shows the improvement

---

## Contributing

Contributions are welcome. Here is how to get started:

### Reporting issues

If you find a bug or unexpected behavior, open an issue on GitHub with:
- Your OS, GPU model, and VRAM
- CUDA version (`nvcc --version`)
- The exact error message and which file it came from
- Steps to reproduce

### Making changes
```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/your-username/morpheus.git
cd morpheus

# 3. Create a branch for your change
git checkout -b feat/your-feature-name

# 4. Make your changes and test them

# 5. Commit with a descriptive message
git commit -m "feat: description of what you changed"

# 6. Push to your fork
git push origin feat/your-feature-name

# 7. Open a pull request on GitHub
```

### Areas open for contribution

- **New model support** — adding support for Llama, Mistral, or other architectures
- **Smarter critic prompts** — improving the quality of Claude's training signal
- **Training strategies** — experimenting with different LoRA ranks, learning rates, or optimizers
- **UI improvements** — enhancing the dashboard with new visualizations
- **Linux/Mac support** — testing and documenting the setup on non-Windows systems
- **Performance** — reducing training latency or improving inference speed

### Code style

- Keep functions small and single-purpose
- Add a comment explaining why, not just what
- Test your change end to end before submitting

---

## License

MIT

---

## Acknowledgements

Built with [Unsloth](https://github.com/unslothai/unsloth), [Claude API](https://anthropic.com), [HuggingFace Transformers](https://huggingface.co/transformers), and [FastAPI](https://fastapi.tiangolo.com).