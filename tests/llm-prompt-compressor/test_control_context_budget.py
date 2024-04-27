import pytest
from llmcontextcompressor.config import settings
from llmcontextcompressor.constants import RankMethodType
from llmcontextcompressor.control_context_budget import control_context_budget


@pytest.mark.asyncio
async def test_control_context_budget():
    # Prepare test data
    context = [
        "Paris is the capital of France.",
        "London is the capital of the United Kingdom.",
        "Baseball is a sport played with a bat and a ball.",
    ]

    question = "What is the capital of France?"
    context_tokens_length = [6, 7, 6]
    target_token = 10
    question = "What is the question?"
    llm_type = RankMethodType.OPEN_AI
    llm_api_config={"open_api_key": settings.OPENAI_API_KEY}
    concurrent = 0
    context_budget = "+5"
    dynamic_context_compression_ratio = 0.0
    reorder_context = "original"



    result, dynamic_ratio, used = await control_context_budget(
        context,
        context_tokens_length,
        target_token,
        question,
        llm_type,
        llm_api_config,
        concurrent,
        context_budget,
        dynamic_context_compression_ratio,
        reorder_context,
    )

    # Assert the expected results
    assert len(result) == 1
    assert result == ["Paris is the capital of France."]
    assert dynamic_ratio == [0.0]
    assert used == [0]
