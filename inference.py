import asyncio
import os
import requests
import json
from typing import List

from openai import OpenAI

# ✅ MUST use these EXACT env variables
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

ENV_URL = "http://localhost:8000"   # local env (validator uses docker)
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 15

# ✅ Correct client (MANDATORY)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


# ---------------- LOGGING (MANDATORY FORMAT) ---------------- #

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


# ---------------- LLM AGENT (IMPORTANT) ---------------- #

def get_action_from_llm(obs):
    prompt = f"""
You are performing data annotation quality control.

Text: {obs['text']}
Given Label: {obs['given_label']}

Decide:
- is_correct (true/false)
- correct_label (positive/negative/neutral)
- confidence (0.0 to 1.0)

Return ONLY valid JSON:
{{
  "is_correct": true/false,
  "correct_label": "positive/negative/neutral",
  "confidence": 0.0-1.0
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )

        content = response.choices[0].message.content.strip()

        # Try parsing JSON
        return json.loads(content)

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)

        # fallback (safe action)
        return {
            "is_correct": True,
            "correct_label": obs["given_label"],
            "confidence": 0.5
        }


# ---------------- MAIN LOOP ---------------- #

async def run_task(task_name):
    rewards = []
    steps = 0
    score = 0.0
    success = False

    log_start(task_name, "qc_env", MODEL_NAME)

    try:
        obs = requests.post(f"{ENV_URL}/reset").json()

        for step in range(1, MAX_STEPS + 1):
            action = get_action_from_llm(obs)

            res = requests.post(f"{ENV_URL}/step", json=action).json()

            reward = res["reward"]["score"]
            done = res["done"]

            rewards.append(reward)
            steps = step

            log_step(step, str(action), reward, done, None)

            if done:
                break

            obs = res["observation"]

        # normalize score
        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score > 0.5

    except Exception as e:
        log_step(steps, "error", 0.0, True, str(e))

    log_end(success, steps, score, rewards)


async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())
