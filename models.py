from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional


class EmailTriageAction(Action):
    action: str = Field(..., description="One of: mark_high, mark_low, reply, ignore, escalate")


class EmailTriageObservation(Observation):
    email_text: str = Field(..., description="Email content")
    email_type: str = Field(..., description="Type: urgent, spam, complaint, info")
    task_type:  str = Field(..., description="Task difficulty: easy, medium, hard")
    done:       bool  = False
    reward:     float = 0.0
    metadata:   dict  = {}