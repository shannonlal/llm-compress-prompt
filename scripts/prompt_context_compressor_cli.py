"""
The following is a command line script used to run the rank results function from the llmpromptcompressor package.
"""

import asyncio
import argparse
from typing import List
from llmcontextcompressor import RankMethodType, settings, get_token_length
from llmcontextcompressor.llm_prompt_compressor import LLMPromptCompressor


async def main(args):
    context = args.context
    question = args.question
    rank_method_type = RankMethodType(args.rank_method)
    concurrent = args.concurrent
    target_token = args.target_token
    llm_api_config = {"open_api_key": settings.OPENAI_API_KEY}

    try:        
        prompt_context_compressor = LLMPromptCompressor(rank_method=rank_method_type, concurrent_requests=concurrent, llm_api_config=llm_api_config)

        original_tokens = sum(get_token_length(rank_method_type, c) for c in context)
        response = await prompt_context_compressor.compress_prompt(
            context=context,
            question=question,
            target_token=int(original_tokens * 0.5)
        )
        print(f"Compressed Prompt: {response}")
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