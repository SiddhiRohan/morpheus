import os
os.environ["TORCHDYNAMO_DISABLE"]            = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"]        = "1"
os.environ["UNSLOTH_DISABLE_CUSTOM_KERNELS"] = "1"

import torch
import threading
import time
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from config import (
    MODEL_NAME, MAX_SEQ_LENGTH, LOAD_IN_4BIT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES
)
from agents.critic  import score_response
from agents.curator import Curator
from training.trainer import MorpheusTrainer as Trainer

model         = None
tokenizer     = None
curator       = Curator()
trainer       = None
swap_lock     = threading.Lock()
train_event   = threading.Event()
stop_event    = threading.Event()
turn_count    = 0
answer_log    = {}


def load_model():
    global model, tokenizer, trainer

    print("[main] Loading model...")
    m, t = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = LOAD_IN_4BIT,
        dtype          = None,
    )
    m = FastLanguageModel.get_peft_model(
        m,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = TARGET_MODULES,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )
    t = get_chat_template(t, chat_template="qwen-2.5")
    m.config.use_cache = True
    FastLanguageModel.for_inference(m)

    model     = m
    tokenizer = t
    trainer   = Trainer(model, tokenizer)
    print("[main] Model ready.")


def generate_response(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,
    )
    inputs = tokenizer(
        text,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = MAX_SEQ_LENGTH,
    )
    input_ids      = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    prompt_len     = input_ids.shape[1]

    with swap_lock:
        with torch.no_grad():
            outputs = model.generate(
                input_ids          = input_ids,
                attention_mask     = attention_mask,
                max_new_tokens     = 200,
                do_sample          = False,
                pad_token_id       = tokenizer.eos_token_id,
                repetition_penalty = 1.15,
                use_cache          = False,
            )

    new_tokens = outputs[0][prompt_len:]
    answer     = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer


def training_loop():
    print("[trainer-thread] Started.")
    while not stop_event.is_set():
        if curator.has_items():
            pair = curator.get_next()
            if pair:
                print(f"[trainer-thread] Training on: '{pair['prompt'][:50]}...'")
                with swap_lock:
                    trainer.train_on_pair(pair["prompt"], pair["ideal_answer"])
                train_event.set()
        else:
            time.sleep(2)
    print("[trainer-thread] Stopped.")


def chat(question: str) -> str:
    global turn_count
    turn_count += 1

    print(f"\n[turn {turn_count}] Generating response...")
    answer = generate_response(question)
    print(f"[turn {turn_count}] Answer: {answer[:100]}...")

    if question not in answer_log:
        answer_log[question] = {"first": answer, "first_turn": turn_count}
    answer_log[question]["latest"] = answer
    answer_log[question]["latest_turn"] = turn_count

    def critique_async():
        print(f"[critic] Scoring turn {turn_count}...")
        result = score_response(question, answer)
        if result:
            print(f"[critic] Score: {result['score']}/10")
            curator.add(question, answer, result)

    thread = threading.Thread(target=critique_async, daemon=True)
    thread.start()
    return answer


def show_comparison(question: str):
    if question not in answer_log:
        print("No data for this question yet.")
        return
    log = answer_log[question]
    print(f"\n{'='*60}")
    print(f"BEFORE (turn {log['first_turn']}):")
    print(log["first"])
    print(f"\nAFTER  (turn {log['latest_turn']}):")
    print(log["latest"])
    print(f"{'='*60}")


def show_stats():
    print(f"\n── Stats ──────────────────────────────")
    print(f"Turns completed  : {turn_count}")
    print(f"Adapter version  : v{trainer.get_adapter_version()}")
    print(f"Loss history     : {trainer.get_loss_history()}")
    cstats = curator.stats()
    print(f"Critic received  : {cstats['total_received']}")
    print(f"Trained on       : {cstats['total_queued']}")
    print(f"Queue now        : {cstats['current_queue']}")
    print(f"───────────────────────────────────────")


if __name__ == "__main__":
    load_model()

    t = threading.Thread(target=training_loop, daemon=True)
    t.start()

    print("\nMorpheus Qwen is running. Type your question.")
    print("Commands: 'stats', 'compare <question>', 'quit'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            show_stats()
            continue
        if user_input.lower().startswith("compare "):
            show_comparison(user_input[8:].strip())
            continue

        answer = chat(user_input)
        print(f"\nMorpheus: {answer}\n")

    stop_event.set()
    print("\n[main] Shutting down.")