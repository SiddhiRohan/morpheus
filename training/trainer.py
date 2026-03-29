import os
os.environ["TORCHDYNAMO_DISABLE"]            = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"]        = "1"
os.environ["UNSLOTH_DISABLE_CUSTOM_KERNELS"] = "1"

import torch
import torch.nn as nn
import json
import time
from datetime import datetime
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel
from datasets import Dataset
from config import (
    TRAIN_BATCH_SIZE, TRAIN_MAX_STEPS,
    LEARNING_RATE, ADAPTER_SAVE_PATH, LOG_DIR
)

# ── Patch Unsloth's broken Triton loss with standard PyTorch ──
import unsloth_zoo.loss_utils as _lu
import unsloth.kernels.cross_entropy_loss as _ce

def _standard_cross_entropy(logits, labels, num_items_in_batch=None,
                             ignore_index=-100, **kwargs):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

_lu.unsloth_fixed_cross_entropy = _standard_cross_entropy
# ─────────────────────────────────────────────────────────────


class MorpheusTrainer:
    """
    Runs micro-batch LoRA updates on (prompt, ideal_answer) pairs.
    Uses standard PyTorch cross entropy loss to avoid Triton kernel
    issues on Windows.
    """

    def __init__(self, model, tokenizer):
        self.model           = model
        self.tokenizer       = tokenizer
        self.adapter_version = 0
        self.loss_history    = []
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(ADAPTER_SAVE_PATH, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

    def _tokenize(self, prompt: str, ideal_answer: str):
        formatted = (
            f"<|im_start|>system\nYou are a concise AI assistant. Answer questions accurately "
            f"and completely in 3-4 short paragraphs maximum. Never cut off mid-sentence. "
            f"Always finish your answer completely within your response.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{ideal_answer}<|im_end|>"
        )
        encodings = self.tokenizer(
            formatted,
            truncation     = True,
            max_length     = 512,
            padding        = False,
            return_tensors = None,
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    def train_on_pair(self, prompt: str, ideal_answer: str) -> dict:
        from unsloth import FastLanguageModel as FLM
        FLM.for_training(self.model)

        if self.loss_history and min(self.loss_history) < 1.5:
            print(f"[trainer] Loss already low — skipping to prevent collapse")
            FLM.for_inference(self.model)
            return {"loss": 0, "adapter_version": self.adapter_version,
                    "duration_sec": 0, "timestamp": datetime.now().isoformat()}

        tokenized = self._tokenize(prompt, ideal_answer)
        dataset   = Dataset.from_dict({
            "input_ids"      : [tokenized["input_ids"]],
            "attention_mask" : [tokenized["attention_mask"]],
            "labels"         : [tokenized["labels"]],
        })

        training_args = TrainingArguments(
            output_dir                  = ADAPTER_SAVE_PATH,
            per_device_train_batch_size = TRAIN_BATCH_SIZE,
            max_steps                   = TRAIN_MAX_STEPS,
            learning_rate               = LEARNING_RATE,
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            logging_steps               = 1,
            save_strategy               = "no",
            report_to                   = "none",
            dataloader_pin_memory       = False,
            remove_unused_columns       = False,
        )

        def data_collator(features):
            return {
                "input_ids"      : torch.tensor([f["input_ids"]      for f in features]),
                "attention_mask" : torch.tensor([f["attention_mask"] for f in features]),
                "labels"         : torch.tensor([f["labels"]         for f in features]),
            }

        hf_trainer = Trainer(
            model         = self.model,
            args          = training_args,
            train_dataset = dataset,
            data_collator = data_collator,
        )

        start        = time.time()
        train_result = hf_trainer.train()
        duration     = time.time() - start

        loss = train_result.training_loss
        self.loss_history.append(round(loss, 4))

        self.adapter_version += 1
        save_path = os.path.join(ADAPTER_SAVE_PATH, f"v{self.adapter_version}")
        self.model.save_pretrained(save_path)

        FLM.for_inference(self.model)

        result = {
            "loss"            : round(loss, 4),
            "adapter_version" : self.adapter_version,
            "duration_sec"    : round(duration, 2),
            "timestamp"       : datetime.now().isoformat(),
        }

        self._log(result)
        print(f"[trainer] v{self.adapter_version} | "
              f"loss: {loss:.4f} | "
              f"took {duration:.1f}s")
        return result

    def _log(self, result: dict):
        log_path = os.path.join(LOG_DIR, "training_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    def get_loss_history(self)    -> list: return self.loss_history
    def get_adapter_version(self) -> int:  return self.adapter_version