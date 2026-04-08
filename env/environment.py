from env.models import Observation, Reward
from env.data import load_dataset
from env.graders import compute_reward


class QCEnvironment:
    def __init__(self, task_type="hard"):
        self.task_type = task_type
        self.data = []
        self.index = 0

    def reset(self):
        self.data = load_dataset()
        self.index = 0
        return self._get_obs()

    def _get_obs(self):
        if self.index >= len(self.data):
            return None

        item = self.data[self.index]

        return Observation(
            text=item["text"],
            given_label=item["given"],
            annotator_id=item["annotator"],
            task_type=self.task_type,
            remaining_steps=len(self.data) - self.index
        )

    def step(self, action):
        item = self.data[self.index]

        score = compute_reward(
            action,
            item["true"],
            item["given"]
        )

        reward = Reward(score=score)

        self.index += 1
        done = self.index >= len(self.data)

        return (
            None if done else self._get_obs(),
            reward,
            done,
            {}
        )

    def state(self):
        return {
            "current_index": self.index,
            "processed": self.index
        }