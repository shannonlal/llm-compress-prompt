
from typing import List

from llmpromptcompressor.constants import RankMethodType
from llmpromptcompressor.rank_results import get_rank_results


async def control_context_budget(        
        context: List[str],
        context_tokens_length: List[int],
        target_token: float,
        question: str,
        llm_type: RankMethodType = RankMethodType.OPEN_AI,
        llm_api_config: dict = None,
        concurrent: int = 0,
        context_budget: str = "+100",
        dynamic_context_compression_ratio: float = 0.0,
        reorder_context: str = "original"
        ):
    """
    The following is the control context budget function which is used to identify contexts that can 
    be removed from the prompt.  It uses the rank method to determine how relevant the context is to the 
    question.  The function then removes the least relevant contexts until the target token count is reached.

    Args:
        context (List[str]): List of context strings that form the basis of the prompt.
        context_tokens_length (List[int]): List of integers representing the number of tokens in each context string.
        target_token (float, optional): The maximum number of tokens to be achieved. Default is -1, indicating no specific target.
                The actual number of tokens after compression should generally be less than the specified target_token, but there can
                be fluctuations due to differences in tokenizers. If specified, compression will be based on the target_token as
                the sole criterion, overriding the ``rate``.
        llm_type (LLMType, optional): The type of LLM to use. Default is LLMType.OPEN_AI.
        question (str, optional): A specific question that the prompt is addressing. Default is an empty string.
        concurrent (int, optional): The number of concurrent requests to make to the LLM API. Default is 0.
        context_budget (str, optional): Token budget for the context-level filtering, expressed as a string to indicate flexibility. Default is "+100".
        dynamic_context_compression_ratio (float, optional): Ratio for dynamically adjusting context compression. Default is 0.0.
        reorder_context: (str, optional): The order in which the context is compressed. Default is "original".
    """
    context_idxs = []
    context_ranked = await get_rank_results(
        context,
        question,
        llm_type,
        concurrent,
        llm_api_config
        )

    if target_token < 0:
        target_token = 100
    target_token = eval("target_token" + context_budget)
    res = []
    used = []

    context_idxs.append([x for idx, (x, _) in enumerate(context_ranked)])
    for idx, _ in context_ranked:
        if idx >= len(context_tokens_length):
            continue
        target_token -= context_tokens_length[idx]
        if idx not in used:
            used.append(idx)
        if target_token <= 0:
            break

    original_used = used
    # Only supporting original for now
    if reorder_context == "original":
        used = sorted(used)

    # Explain how dynamic context compression works?
    if dynamic_context_compression_ratio > 0:
        N = len(used)
        dynamic_ratio = [
            i * (abs(dynamic_context_compression_ratio) / (N - 1)) if N > 1 else 0
            for i in range(-(N - 1), N, 2)
        ][::-1]
        dynamic_ratio_map = {i: j for i, j in zip(original_used, dynamic_ratio)}
        dynamic_ratio = [dynamic_ratio_map[i] for i in used]
    else:
        dynamic_ratio = [0.0] * len(used)


    res = [context[idx] for idx in used if idx < len(context)]
    return res, dynamic_ratio, used