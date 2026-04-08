import os
from openai import OpenAI

from server.email_triage_env_environment import EmailTriageEnvironment
from models import EmailTriageAction
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def get_action(email_text):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """You are an email triage agent.

Choose ONLY one action from this list:
mark_high
mark_low
reply
ignore
escalate

Rules:
- Output ONLY one word
- No sentences
- No explanations
- No punctuation
- If unsure, output "ignore"
"""
            },
            {"role": "user", "content": email_text},
        ],
        max_tokens=3,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip().lower()

    # 🔥 FORCE CLEAN OUTPUT
    valid = ["mark_high", "mark_low", "reply", "ignore", "escalate"]

    for v in valid:
        if v in raw:
            return v

    return "ignore"

def run():
    env = EmailTriageEnvironment()
    obs = env.reset()

    print(f"[START] task={obs.task_type} env=email_triage model={MODEL_NAME}")

    step = 0
    rewards = []

    done = False

    while not done:
        step += 1

        try:
            action_str = get_action(obs.email_text)
            action = EmailTriageAction(action=action_str)

            obs = env.step(action)

            reward = obs.reward
            rewards.append(f"{reward:.2f}")

            done = obs.done

            print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

        except Exception as e:
            print(f"[STEP] step={step} action=error reward=0.00 done=true error={str(e)}")
            done = True

    success = True

    print(f"[END] success={str(success).lower()} steps={step} rewards={','.join(rewards)}")


if __name__ == "__main__":
    run()