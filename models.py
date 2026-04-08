from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EmailTriageAction(Action):
    action: str = Field(..., description="Action taken by agent")


class EmailTriageObservation(Observation):
    email_text: str = Field(..., description="Email content")
    email_type: str = Field(..., description="Type of email")
    task_type: str = Field(..., description="Task difficulty")

    done: bool = False
    reward: float = 0.0
    metadata: dict = {}