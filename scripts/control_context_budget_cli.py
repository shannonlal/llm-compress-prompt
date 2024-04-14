"""
The following is a command line script used to run the rank results function from the llmpromptcompressor package.
"""

import asyncio
import argparse
from typing import List
from llmpromptcompressor import RankMethodType, control_context_budget, settings, get_token_length


async def main(args):
    context = args.context
    question = args.question
    rank_method_type = RankMethodType(args.rank_method)
    concurrent = args.concurrent
    target_token = args.target_token
    llm_api_config = {"open_api_key": settings.OPENAI_API_KEY}

    try:
        context_tokens_length = []
        for context_str in context:
            context_tokens_length.append(get_token_length(rank_method_type, context_str))

        context, dynamic_ratio, context_used = await control_context_budget(
            context, 
            context_tokens_length, 
            target_token,
            rank_method_type,
            llm_api_config,
            question,
            concurrent )
        print(f"Control Context Budet: {context}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank context based on a question using LLM.")
    parser.add_argument("--context", nargs="+", required=True, help="List of context strings")
    parser.add_argument("--question", type=str, required=True, help="Question string")
    parser.add_argument("--rank_method", type=str, required=True, help="LLM type (e.g., OPEN_AI)")
    parser.add_argument("--concurrent", type=int, default=0, help="Number of concurrent requests (default: 0)")
    parser.add_argument("--target_token", type=float, default=100, help="Target token count (default: 100)")

    args = parser.parse_args()

    asyncio.run(main(args))