import pytest
from  llmpromptcompressor import get_token_length, RankMethodType

def test_unsupported_llm():
    with pytest.raises(NotImplementedError, match="Model not supported"):
        get_token_length("UNSUPPORTED_LLM", "Sample prompt")

def test_none_prompt():
    assert get_token_length("OPEN_AI", None) == 0

def test_empty_prompt():
    assert get_token_length("OPEN_AI", "") == 0

def test_valid_prompt():
    prompt = "This is a sample prompt."
    expected_token_length = 6  # Assuming the prompt has 6 tokens
    assert get_token_length(RankMethodType.OPEN_AI, prompt) == expected_token_length