import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import EmailTriageAction, EmailTriageObservation


class EmailTriageEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_type = "easy" 

    def reset(self, task: str = "easy") -> EmailTriageObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.task_type = task

        self.emails = [
            {"type": "urgent", "text": "Server down"},
            {"type": "spam", "text": "Win money"},
            {"type": "complaint", "text": "Product broken"},
            {"type": "info", "text": "Weekly update"},
        ]

        random.shuffle(self.emails)

        self.index = 0
        self.correct_actions = 0
        self.total_actions = 0

        email = self.emails[self.index]

        return EmailTriageObservation(
            email_text=email["text"],
            email_type=email["type"],
            task_type=self.task_type,
            done=False,
            reward=0.0,
        )

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        if not hasattr(self, "emails"):
            self.reset(task=self.task_type)

        self._state.step_count += 1

        valid_actions = ["mark_high", "mark_low", "reply", "ignore", "escalate"]
        if action.action not in valid_actions:
            action.action = "ignore"

        email = self.emails[self.index]
        correct = False
        reward = 0.0
        if self.task_type == "easy":
            correct = (
                (email["type"] in ["urgent", "complaint"] and action.action == "mark_high")
                or (email["type"] in ["spam", "info"] and action.action == "mark_low")
            )
        elif self.task_type == "medium":
            if email["type"] == "urgent":
                correct = action.action in ["reply", "escalate"]
            elif email["type"] == "spam":
                correct = action.action == "ignore"
            elif email["type"] == "complaint":
                correct = action.action in ["reply", "escalate"]
            else:
                correct = action.action == "mark_low"
        elif self.task_type == "hard":
            if email["type"] == "urgent":
                correct = action.action in ["reply", "escalate"]
            elif email["type"] == "spam":
                correct = action.action == "ignore"
            elif email["type"] == "complaint":
                correct = action.action == "reply"
            else:
                correct = action.action == "mark_low"

        # reward
        if correct:
            reward += 1.0
            self.correct_actions += 1
        else:
            reward -= 1.0

        if self.task_type == "hard":
            reward -= 0.1

        self.total_actions += 1

        self.index += 1
        done = self.index >= len(self.emails)

        next_email = self.emails[self.index] if not done else {"text": "", "type": ""}

        return EmailTriageObservation(
            email_text=next_email["text"],
            email_type=next_email["type"],
            task_type=self.task_type,
            done=done,
            reward=reward,
            metadata={
                "task": self.task_type,   
                "correct": correct,
            },
        )
    def compute_score(self, task: str = None) -> float:
        if self.total_actions == 0:
            return 0.5

        raw = self.correct_actions / self.total_actions

        if task == "easy":
            score = 0.10 + raw * 0.88      
        elif task == "medium":
            score = 0.08 + raw * 0.88      
        elif task == "hard":
            penalty = 0.1 * self.total_actions
            adjusted = max(0, self.correct_actions - penalty)
            raw_hard = adjusted / self.total_actions
            score = 0.05 + raw_hard * 0.88  
        else:
            score = 0.10 + raw * 0.88

        return max(0.01, min(0.99, score))  

    @property
    def state(self):
        return self._state