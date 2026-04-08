from pydantic import BaseModel, Field


class Observation(BaseModel):
    text: str
    given_label: str
    annotator_id: str
    task_type: str
    remaining_steps: int


class Action(BaseModel):
    is_correct: bool
    correct_label: str
    confidence: float = Field(ge=0.0, le=1.0)


class Reward(BaseModel):
    score: float