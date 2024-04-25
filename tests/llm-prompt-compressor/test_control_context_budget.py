# from typing import List
# from unittest.mock import AsyncMock, patch

# import pytest

# from llmpromptcompressor.constants import RankMethodType
# from llmpromptcompressor.control_context_budget import control_context_budget


# @pytest.mark.asyncio
# async def test_control_context_budget():
#     # Prepare test data
#     context = [
#         "The quick brown fox jumps over the lazy dog.",
#         "The lazy dog sleeps all day long.",
#     ]
#     context_tokens_length = [9, 7]
#     target_token = 8
#     question = "What does the quick brown fox do?"

#     # Mock the get_rank_results function
#     with patch("llmpromptcompressor.get_rank_results", new_callable=AsyncMock) as mock_get_rank_results:
#         mock_get_rank_results.return_value = [(0, 0.8), (1, 0.2)]

#         # Call the function
#         result, dynamic_ratio, used = await control_context_budget(
#             context,
#             context_tokens_length,
#             target_token,
#             question,
#             llm_type=RankMethodType.OPEN_AI,
#             llm_api_config=None,
#             concurrent=0,
#             context_budget="+100",
#             dynamic_context_compression_ratio=0.0,
#             reorder_context="original",
#         )

#         # Assert the expected results
#         assert result == ["The quick brown fox jumps over the lazy dog."]
#         assert dynamic_ratio == [0.0]
#         assert used == [0]

#         # Assert that get_rank_results was called with the correct arguments
#         mock_get_rank_results.assert_called_once_with(
#             context,
#             question,
#             RankMethodType.OPEN_AI,
#             0,
#             None,
#         )