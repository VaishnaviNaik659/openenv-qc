import asyncio
import os
import requests
from typing import List

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

ENV_URL = "http://localhost:8000"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 15

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


# ✅ Deterministic high-score agent
def get_action(obs):
    text = obs["text"].lower()
    given = obs["given_label"]

    if any(word in text for word in ["love", "amazing", "great", "fantastic", "good"]):
        true_label = "positive"
    elif any(word in text for word in ["hate", "worst", "terrible", "bad"]):
        true_label = "negative"
    else:
        true_label = "neutral"

    is_correct = (given == true_label)

    return {
        "is_correct": is_correct,
        "correct_label": true_label,
        "confidence": 0.9 if is_correct else 0.8
    }


async def run_task(task_name):
    rewards = []
    steps = 0

    log_start(task_name, "qc_env", MODEL_NAME)

    try:
        obs = requests.post(f"{ENV_URL}/reset").json()

        for step in range(1, MAX_STEPS + 1):
            action = get_action(obs)

            res = requests.post(f"{ENV_URL}/step", json=action).json()

            reward = res["reward"]["score"]
            done = res["done"]

            rewards.append(reward)
            steps = step

            log_step(step, str(action), reward, done, None)

            if done:
                break

            obs = res["observation"]

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score > 0.6

    except Exception as e:
        log_step(steps, "error", 0.0, True, str(e))
        score = 0.0
        success = False

    log_end(success, steps, score, rewards)


async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())