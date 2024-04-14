import pytest
import asyncio
from llmpromptcompressor.constants import RankMethodType
from  llmpromptcompressor.rank_results import get_rank_results

@pytest.mark.asyncio
async def test_empty_context():
    context = []
    question = "What is the capital of France?"
    llm_type = RankMethodType.OPEN_AI
    with pytest.raises(IndexError):
        await get_rank_results(context, question, llm_type)

@pytest.mark.asyncio
async def test_empty_question():
    context = ["Paris is the capital of France."]
    question = ""
    llm_type = RankMethodType.OPEN_AI
    with pytest.raises(ValueError):
        await get_rank_results(context, question, llm_type)

@pytest.mark.asyncio
async def test_unsupported_llm_type():
    context = ["Paris is the capital of France."]
    question = "What is the capital of France?"
    llm_type = "UNSUPPORTED_LLM"
    with pytest.raises(NotImplementedError, match="Rank method not supported"):
        await get_rank_results(context, question, llm_type)

@pytest.mark.asyncio
async def test_single_context_single_question():
    context = ["Paris is the capital of France."]
    question = "What is the capital of France?"
    llm_type = RankMethodType.OPEN_AI
    llm_api_config = {"open_api_key": "your_api_key"}
    result = await get_rank_results(context, question, llm_type, llm_api_config=llm_api_config)
    assert len(result) == 1
    assert result[0][0] == 0

@pytest.mark.asyncio
async def test_multiple_context_single_question():
    context = [
        "Paris is the capital of France.",
        "London is the capital of the United Kingdom.",
        "Berlin is the capital of Germany.",
        "Madrid is the capital of Spain.",
        "Rome is the capital of Italy.",
        "Tokyo is the capital of Japan.",
        "Beijing is the capital of China.",
        "Moscow is the capital of Russia.",
        "Washington D.C. is the capital of the United States.",
        "Canberra is the capital of Australia."
    ]
    question = "What is the capital of France?"
    llm_type = RankMethodType.OPEN_AI
    llm_api_config = {"open_api_key": "your_api_key"}
    result = await get_rank_results(context, question, llm_type, llm_api_config=llm_api_config)
    assert len(result) == 10
    assert result[0][0] == 0

@pytest.mark.asyncio
async def test_multiple_context_single_question_concurrent():
    context = [
        "Paris is the capital of France.",
        "London is the capital of the United Kingdom.",
        "Berlin is the capital of Germany.",
        "Madrid is the capital of Spain.",
        "Rome is the capital of Italy.",
        "Tokyo is the capital of Japan.",
        "Beijing is the capital of China.",
        "Moscow is the capital of Russia.",
        "Washington D.C. is the capital of the United States.",
        "Canberra is the capital of Australia."
    ]
    question = "What is the capital of France?"
    llm_type = RankMethodType.OPEN_AI
    concurrent = 3
    llm_api_config = {"open_api_key": "your_api_key"}
    result = await get_rank_results(context, question, llm_type, concurrent=concurrent, llm_api_config=llm_api_config)
    assert len(result) == 10
    assert result[0][0] == 0