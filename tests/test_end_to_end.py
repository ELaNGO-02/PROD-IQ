# from ml_core.pipeline import run_full_pipeline
from ml_core.pipeline import run_full_pipeline

def test_full_pipeline_success():
    user_input = """
    We are building a SaaS product for small startups.
    3 founders, based in India.
    Raised $50k so far.
    Subscription model.
    """

    result = run_full_pipeline(user_input, task="success")

    print("\nPIPELINE OUTPUT:\n", result)

    assert "probability" in result
