"""
The following is a command line script used to run the rank results function from the llmpromptcompressor package.
"""

import asyncio
import argparse
from typing import List
from llmcontextcompressor import RankMethodType, get_rank_results, settings


async def main(args):
    context = args.context
    question = args.question
    rank_method = RankMethodType(args.rank_method)
    concurrent = args.concurrent
    llm_api_config = {"open_api_key": settings.OPENAI_API_KEY}

    try:
        result = await get_rank_results(context, question, rank_method, concurrent, llm_api_config)
        print("Ranking Results:")
        for idx, score in result:
            print(f"Index: {idx}, Score: {score}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank context based on a question using LLM.")
    parser.add_argument("--context", nargs="+", required=True, help="List of context strings")
    parser.add_argument("--question", type=str, required=True, help="Question string")
    parser.add_argument("--rank_method", type=str, required=True, help="Rank Method type (e.g., OPEN_AI)")
    parser.add_argument("--concurrent", type=int, default=0, help="Number of concurrent requests (default: 0)")

    args = parser.parse_args()

    asyncio.run(main(args))