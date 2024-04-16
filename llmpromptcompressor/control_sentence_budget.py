from typing import List
from llmpromptcompressor.constants import RankMethodType
from llmpromptcompressor.rank_results import get_rank_results
from llmpromptcompressor.token import get_token_length
import nltk
from collections import defaultdict

async def control_sentence_budget(context: List[str],
                            question: str,
                            target_token: float,
                            rank_method_type: RankMethodType,
                            concurrent: int = 0,
                            llm_api_config: dict = None,
                            token_budget_ratio: float = 1.4,):
    """
    Control the sentence budget will remove the least relevant sentences from the context until the target token count is reached.

    Args:
        context (List[str]): List of context strings that form the basis of the prompt.
        question (str): A specific question that the prompt is addressing.
        target_token (float): The maximum number of tokens to be achieved.
        rank_method_type (RankMethodType): The type of LLM to use.
        concurrent (int, optional): The number of concurrent requests to make to the LLM API. Default is 0.
        llm_api_config (dict, optional): The configuration for the LLM API. Default is None.
        token_budget_ratio (float, optional): The ratio of the token budget. Default is 1.4.
    """
    sentences = [nltk.sent_tokenize(c) for c in context]
    dem_g, s2de, idx = defaultdict(set), defaultdict(int), 0
    for idx_d, s in enumerate(sentences):
        for _ in s:
            dem_g[idx_d].add(idx)
            s2de[idx] = idx_d
            idx += 1

    context_sentences = [s for ii in sentences for s in ii]
    sentence_tokens_length = [
        get_token_length(rank_method_type, sentence) for sentence in context_sentences
    ]
    N = len(context_sentences)
    flags = list(range(len(context_sentences)))
    if len(sentence_tokens_length) == 1:
        return context

    sent_sort = await get_rank_results(
        context_sentences,
        question,
        rank_method_type,
        concurrent,
        llm_api_config,
    )

    sentence_flags = [False] * N
    if target_token < 0:
        target_token = 100
    target_token *= token_budget_ratio
    res = []
    for idx, _ in sent_sort:
        idx = flags[idx]
        target_token -= sentence_tokens_length[idx]
        sentence_flags[idx] = True
        if target_token < 0:
            break


    idx = 0
    res = []
    new_segments_info = []
    for s in sentences:
        tmp = [jj for ii, jj in enumerate(s) if sentence_flags[idx + ii]]
        res.append("".join(tmp))
        idx += len(s)

    return res, new_segments_info