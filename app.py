from fastapi import FastAPI
from env.environment import QCEnvironment
from env.models import Action

app = FastAPI()

env = QCEnvironment(task_type="hard")


@app.post("/reset")   # ✅ FIXED (was GET)
def reset():
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.dict() if obs else None,
        "reward": reward.dict(),
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env.state()