import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# Config
# -------------------------
MODEL_ID = "prod-IQ/prod-iq-quantized-4bit"
GLOBAL_PROMPT_PATH = Path("prompts/global_prompt.txt")

MAX_NEW_TOKENS = 48
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.05

# -------------------------
# Load Global Prompt
# -------------------------
def load_global_prompt():
    if not GLOBAL_PROMPT_PATH.exists():
        raise FileNotFoundError("global_prompt.txt not found")
    return GLOBAL_PROMPT_PATH.read_text(encoding="utf-8")


# -------------------------
# Load Model
# -------------------------
def load_llm():
    print("[TEST] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("[TEST] Loading quantized model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model.eval()

    print("[TEST] Model loaded:", model.config._name_or_path)
    print("[TEST] Layer type:",
          type(model.model.layers[0].self_attn.q_proj))

    return model, tokenizer

# -------------------------
# Inference
# -------------------------
def run_inference(model, tokenizer, prompt: str):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response

# -------------------------
# Test Case
# -------------------------
def test_llm_dry_run():
    torch.set_num_threads(4)

    global_prompt = load_global_prompt()
    model, tokenizer = load_llm()

    user_input = (
        "I am planning a SaaS startup with 3 founders and $50k funding. "
        "What should I focus on in the first 6 months?"
    )

    full_prompt = f"""
{global_prompt}

User:
{user_input}

Assistant:
"""

    print("\n[TEST] Running dry inference...\n")
    response = run_inference(model, tokenizer, full_prompt)

    print("\n[LLM OUTPUT]")
    print("-" * 80)
    print(response)
    print("-" * 80)

    # Smoke test only (no strict assertion)
    assert len(response) > 0, "LLM returned empty output"
