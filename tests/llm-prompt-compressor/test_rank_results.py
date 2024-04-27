import asyncio

import pytest

from llmpromptcompressor import RankMethodType, get_rank_results, settings

@pytest.fixture
def rank_results():
    return get_rank_results

@pytest.mark.asyncio
async def test_get_rank_results_empty_context(rank_results):
    with pytest.raises(ValueError, match="Context cannot be empty"):
        await rank_results([], "What is the question?", RankMethodType.OPEN_AI)

@pytest.mark.asyncio
async def test_get_rank_results_empty_question(rank_results):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        await rank_results(["Some context"], "", RankMethodType.OPEN_AI)

@pytest.mark.asyncio
async def test_get_rank_results_openai_mocked(rank_results):
    context = [
        "The quick brown fox jumps over the lazy dog.",
        "The lazy dog sleeps all day long.",
    ]
    question = "What does the quick brown fox do?"

    result = await rank_results(
                context,
                question,
                RankMethodType.OPEN_AI,
                concurrent=1, 
                llm_api_config={"open_api_key": settings.OPENAI_API_KEY})
    
    assert len(result) == 2
    assert result[0][1] > 0.7
    assert result[0][1] > 0.2

@pytest.mark.asyncio
async def test_get_rank_results_openai_low_rank(rank_results):
    context = [
        "The quick brown fox jumps over the lazy dog.",
        "The lazy dog sleeps all day long.",
    ]
    question = "What is the capital of Paris?"

    result = await rank_results(
                context,
                question,
                RankMethodType.OPEN_AI,
                concurrent=1, 
                llm_api_config={"open_api_key": settings.OPENAI_API_KEY})
    
    assert len(result) == 2
    assert result[0][1] < 0.1

@pytest.mark.asyncio
async def test_get_rank_results_bm25(rank_results):
    context = [
        "Paris is the capital of France.",
        "London is the capital of the United Kingdom.",
        "Baseball is a sport played with a bat and a ball.",
    ]
    question = "What is the capital of France?"

    result = await rank_results(
                context,
                question,
                RankMethodType.BM25,
                concurrent=1, 
                llm_api_config={"open_api_key": ""})
    print(result)
    assert len(result) == 3
    assert result[0][1] > 0.6
    assert result[1][1] > 0.04
    assert result[2][1] > 0.01