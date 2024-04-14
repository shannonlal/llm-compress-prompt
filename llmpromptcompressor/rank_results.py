import asyncio
from typing import List
from .constants import RankMethodType
from sentence_transformers import util
import numpy as np
from numpy.linalg import norm
import aiohttp
from nltk.tokenize import word_tokenize
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

OPENAI_MAX_CONCURRENT = 50

async def get_rank_results(
    context: list,
    question: str,
    rank_method_type: RankMethodType,
    concurrent: int = 0,
    llm_api_config: dict = None,
):
    
    async def get_distance_bm25(corpus, query):
        from rank_bm25 import BM25Okapi

        async def tokenize_document(doc):
            return word_tokenize(doc)
        
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor() as executor:
            tokenized_corpus = await asyncio.gather(
                *[loop.run_in_executor(executor, word_tokenize, doc) for doc in corpus]
            )

        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = word_tokenize(query)

        doc_scores = bm25.get_scores(tokenized_query)
        idx = [(ii, score) for ii, score in zip((-doc_scores).argsort(), doc_scores)]

        return idx
    
    async def get_distance_openai(corpus:List[str], query):

        async def get_embed(text:str):
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {llm_api_config.get('open_api_key', '')}",
                }
                async with session.post(
                    "https://api.openai.com/v1/embeddings",
                    json={"input": text, "model":"text-embedding-3-small"},
                    headers=headers,
                ) as response:
                    data = await response.json()
                    return data["data"][0]["embedding"]

        doc_embeds = []

        if concurrent > 0:
            limit = min(concurrent, OPENAI_MAX_CONCURRENT)
            tasks = []

            for i in corpus:
                task = get_embed(i)
                tasks.append(task)
                if len(tasks) >= limit:
                    responses = await asyncio.gather(*tasks)
                    for response in responses:
                        doc_embeds.append(response)
                    tasks = []
            if tasks:
                responses = await asyncio.gather(*tasks)
                for response in responses:
                    doc_embeds.append(response)
        else:
            doc_embeds = await get_embed(corpus)

        query = await get_embed([query])
        doc_scores = util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
        idx = [(i, score) for i, score in enumerate(doc_scores)]
        idx.sort(key=lambda x: x[1], reverse=True)

        return idx

    method = None
    if rank_method_type == RankMethodType.OPEN_AI:
        method = get_distance_openai
    elif rank_method_type == RankMethodType.BM25:
        method = get_distance_bm25
    else:
        raise NotImplementedError("Rank method not supported")
    return await method(context, question)