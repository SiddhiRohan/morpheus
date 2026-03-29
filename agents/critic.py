import os
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import json
import anthropic
from config import ANTHROPIC_API_KEY, CRITIC_MODEL

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

CRITIC_SYSTEM_PROMPT = """You are an expert AI response evaluator.

Your job is to evaluate a model's answer to a question and return a JSON object.

You must ALWAYS return valid JSON and nothing else. No preamble, no explanation outside the JSON.

Return exactly this structure:
{
  "score": <integer 0-10>,
  "what_was_wrong": "<string describing specific errors, or 'none' if score is 8+>",
  "ideal_answer": "<string with the complete correct answer>"
}

Scoring guide:
0-3  : Completely wrong or dangerously misleading
4-6  : Partially correct but missing key information or contains errors
7    : Mostly correct with minor gaps
8-10 : Accurate, complete, and well explained

Be strict. A score of 7+ means the answer is good enough that training on it would not help much.
Only scores below 7 are worth training on.

CRITICAL LENGTH LIMITS — you must follow these exactly:
- ideal_answer: under 120 words, plain prose, no markdown, no headers, no bullet points
- what_was_wrong: under 60 words, specific and brief
The model being trained outputs maximum 200 tokens. If ideal_answer
exceeds 120 words the training signal becomes unreachable and useless.
Shorter is always better here."""


def score_response(question: str, answer: str) -> dict:
    """
    Send a (question, answer) pair to Claude for evaluation.
    Returns a dict with score, what_was_wrong, ideal_answer.
    Returns None if the API call fails.
    """
    user_message = f"""Question: {question}

Model's answer: {answer}

Evaluate this answer and return your JSON assessment."""

    try:
        response = client.messages.create(
            model      = CRITIC_MODEL,
            max_tokens = 1024,
            system     = CRITIC_SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_message}]
        )

        raw = response.content[0].text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        assert "score"           in result, "missing score"
        assert "what_was_wrong"  in result, "missing what_was_wrong"
        assert "ideal_answer"    in result, "missing ideal_answer"
        assert isinstance(result["score"], int), "score must be int"
        assert 0 <= result["score"] <= 10, "score out of range"

        return result

    except json.JSONDecodeError as e:
        print(f"[critic] JSON parse error: {e}")
        print(f"[critic] Raw response was: {raw}")
        return None

    except AssertionError as e:
        print(f"[critic] Validation error: {e}")
        return None

    except Exception as e:
        print(f"[critic] API error: {e}")
        return None