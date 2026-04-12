import asyncio
import os
import json
from typing import List

from openai import OpenAI
from openenv import OpenEnv

# ENV VARS (MANDATORY)
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

IMAGE_NAME = os.getenv("IMAGE_NAME", "openenv-qc")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 10


# ---------------- LOGGING ---------------- #

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


# ---------------- LLM ---------------- #

def get_action(obs):
    prompt = f"""
Text: {obs.text}
Given Label: {obs.given_label}

Return JSON:
{{
  "is_correct": true/false,
  "correct_label": "positive/negative/neutral",
  "confidence": 0.0
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=100
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except:
        return {
            "is_correct": True,
            "correct_label": obs.given_label,
            "confidence": 0.5
        }


# ---------------- MAIN ---------------- #

async def run_task(task):
    env = await OpenEnv.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps = 0

    log_start(task, "qc_env", MODEL_NAME)

    try:
        result = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            action = get_action(obs)

            result = await env.step(action)

            reward = result.reward.score
            done = result.done

            rewards.append(reward)
            steps = step

            log_step(step, str(action), reward, done, None)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score > 0.5

    finally:
        await env.close()
        log_end(success, steps, score, rewards)


async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())
