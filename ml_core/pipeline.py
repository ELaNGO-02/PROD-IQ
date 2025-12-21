from llm.router import route_task
from llm.extractor import extract_fields
from llm.executor import execute_task
from ml_core.feature_engine import engineer_features

from ml_core.normalizer import normalize_features

MODEL_REQUIREMENTS = {
    "predict_survival": {"total_funding", "burn_rate_monthly"},
    "predict_breakeven": {"total_funding", "burn_rate_monthly"},
    "predict_revenue": {"price"},
    "predict_success": {"main_category", "team_size"},
    "predict_traction": {"main_category"}
}


def run_full_pipeline_llm(user_text: str) -> dict:
    tasks = route_task(user_text)
    extracted = extract_fields(user_text)

    # Feature engineering only ONCE
    features_df = engineer_features(extracted)

    results = {}

    for task in tasks:
        guard = validate_model_inputs(task, extracted)

        if guard:
            results[task] = guard
            continue

        results[task] = execute_task(task, extracted)

    return {
        "tasks": tasks,
        "extracted": extracted,
        "results": results
    }


def validate_model_inputs(task: str, extracted: dict):
    required = MODEL_REQUIREMENTS.get(task, set())
    missing = [k for k in required if not extracted.get(k)]

    if missing:
        return {
            "status": "needs_clarification",
            "task": task,
            "missing_fields": missing,
            "message": f"I need {missing} to answer this accurately."
        }

    return None
