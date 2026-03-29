import os
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import json
import threading
from collections import deque
from config import CURATOR_QUEUE_MAX, SCORE_THRESHOLD, LOG_DIR


class Curator:
    """
    Receives scored (question, answer) pairs from the critic.
    Filters out good answers — only queues pairs the model got wrong.
    Training pulls from this queue.
    """

    def __init__(self):
        self.queue          = deque(maxlen=CURATOR_QUEUE_MAX)
        self.lock           = threading.Lock()
        self.total_received = 0
        self.total_queued   = 0
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        os.makedirs(LOG_DIR, exist_ok=True)

    def add(self, question: str, model_answer: str, critic_result: dict) -> bool:
        """
        Add a scored pair to the queue if it's below the score threshold.
        Returns True if queued, False if skipped (score too high).
        """
        self.total_received += 1
        score = critic_result.get("score", 10)

        if score >= SCORE_THRESHOLD:
            print(f"[curator] Score {score}/10 — skipping (model did well enough)")
            return False

        training_pair = {
            "prompt"        : question,
            "ideal_answer"  : critic_result["ideal_answer"],
            "score"         : score,
            "what_was_wrong": critic_result["what_was_wrong"],
        }

        with self.lock:
            self.queue.append(training_pair)
            self.total_queued += 1

        print(f"[curator] Score {score}/10 — queued for training "
              f"(queue size: {len(self.queue)})")
        self._log(training_pair)
        return True

    def get_next(self) -> dict | None:
        """
        Pop the next training pair from the queue.
        Returns None if queue is empty.
        """
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            return None

    def has_items(self) -> bool:
        with self.lock:
            return len(self.queue) > 0

    def queue_size(self) -> int:
        with self.lock:
            return len(self.queue)

    def stats(self) -> dict:
        return {
            "total_received" : self.total_received,
            "total_queued"   : self.total_queued,
            "current_queue"  : self.queue_size(),
        }

    def _log(self, pair: dict):
        log_path = os.path.join(LOG_DIR, "curator_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair) + "\n")