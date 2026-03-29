import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME     = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT   = True

LORA_R         = 16
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

TRAIN_BATCH_SIZE  = 1
TRAIN_MAX_STEPS   = 2
LEARNING_RATE     = 1e-4
ADAPTER_SAVE_PATH = "model/adapter"

CURATOR_QUEUE_MAX = 50
SCORE_THRESHOLD   = 7

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CRITIC_MODEL      = "claude-sonnet-4-6"

LOG_DIR = "logs"