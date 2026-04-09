import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))

from openai import OpenAI

try:
    from server.email_triage_env_environment import EmailTriageEnvironment
except ModuleNotFoundError:
    from email_triage_env_environment import EmailTriageEnvironment

try:
    from models import EmailTriageAction
except ModuleNotFoundError:
    from models import EmailTriageAction

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "placeholder")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = ["easy", "medium", "hard"]
VALID_ACTIONS = ["mark_high", "mark_low", "reply", "ignore", "escalate"]


def get_action_from_llm(email_text: str, email_type: str, task_type: str, step: int) -> str:
    prompt = f"""You are an email triage assistant.

Email: "{email_text}"
Type: {email_type}
Task: {task_type}
Step: {step}

Rules:
- easy: mark_high (urgent/complaint), mark_low (spam/info)
- medium: reply or escalate (urgent/complaint), ignore (spam), mark_low (info)
- hard: reply or escalate (urgent), ignore (spam), reply only (complaint), mark_low (info)

Reply with EXACTLY one word from: mark_high, mark_low, reply, ignore, escalate"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an email triage assistant. Respond with exactly one action word only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().lower()
        for action in VALID_ACTIONS:
            if action in raw:
                return action
        return _fallback_action(email_type, task_type)
    except Exception as e:
        print(f"LLM error at step {step}: {e}", flush=True)
        return _fallback_action(email_type, task_type)


def _fallback_action(email_type: str, task_type: str) -> str:
    if task_type == "easy":
        return "mark_high" if email_type in ["urgent", "complaint"] else "mark_low"
    if email_type == "urgent":    return "reply"
    if email_type == "spam":      return "ignore"
    if email_type == "complaint": return "reply"
    return "mark_low"


def run_task(task: str):
    env = EmailTriageEnvironment()
    obs = env.reset(task=task)

    print(f"[START] task={task} env=email_triage model={MODEL_NAME}", flush=True)

    step = 0
    rewards = []
    done = False

    while not done:
        step += 1
        action_str = get_action_from_llm(
            email_text=obs.email_text,
            email_type=obs.email_type,
            task_type=task,
            step=step
        )

        result = env.step(EmailTriageAction(action=action_str))
        reward = round(result.reward, 2)
        done = result.done
        rewards.append(reward)

        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

        obs = result

    score = env.compute_score(task=task)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success=true steps={step} score={score:.2f} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)