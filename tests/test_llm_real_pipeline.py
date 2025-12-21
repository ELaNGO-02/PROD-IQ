from ml_core.pipeline import run_full_pipeline_llm

def test_real_llm_pipeline():
    user_input = """We have $80k left, burn is $12k/month.
    Will we survive and when do we break even?
    """

    result = run_full_pipeline_llm(user_input)

    print("\nFULL RESULT:\n", result)

    assert "task" in result
    assert "result" in result
