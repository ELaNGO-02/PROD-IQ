import json
from llm.local_inference import generate

ROUTER_PROMPT = """
Decide what the user wants help with.

Choose ONE word from:
success
revenue
breakeven
survival
traction
market
story

Respond with ONLY the word.
No explanations.

User:
"""

import json
from llm.local_inference import generate
def route_task(user_text: str) -> list[str]:
    response = generate(ROUTER_PROMPT + user_text, max_tokens=20).lower()

    tasks = []

    if "surviv" in response or "runway" in response:
        tasks.append("predict_survival")
    if "break" in response or "profit" in response:
        tasks.append("predict_breakeven")
    if "revenue" in response or "mrr" in response:
        tasks.append("predict_revenue")

    return tasks or ["story_weaver"]

 