import asyncio
import json
import tiktoken
import time
from llmpromptcompressor import LLMPromptCompressor, settings, RankMethodType

from evaluation_utils import  compare_prompts, compare_responses
from prompt_format import get_original_prompt, process_jsonl_file,compress_prompt

## Adjustments to scenario
## 1. Run within a loop and actually get the average time - Done
## 2. Do this with medium set as well - Done
## 3. Evaluate Model difference with GPT3.5-16k - Done
## 4. Try different Compression Size Comparison to see the difference.

## Run Code on LLM Compress with different scenarios and see speed difference
##

async def analyze_prompt( data, target_token):
    start_time = time.time()
    compressed_prompt = await compress_prompt(data, target_token)
    end_time = time.time()
    execution_time = end_time - start_time

    origin_tokens = compressed_prompt["origin_tokens"]
    compressed_tokens = compressed_prompt["compressed_tokens"]
    compressed_prompt = compressed_prompt["compressed_prompt"]
    compressed_ratio = compressed_tokens / origin_tokens
    print(f"Execution Time: {execution_time:.3f} seconds for original tokens {origin_tokens}, Compressed Tokens {compressed_tokens} ")
    print(f"Average Compression Ratio: {compressed_ratio:.2f}")

    ## Evaluate Prompt Similarity
    num_tokens, original_prompt = get_original_prompt(data)

    compare_prompts_start = time.time()
    similarity = await compare_prompts(original_prompt, compressed_prompt)
    print("Compare Prompt Time: ", time.time() - compare_prompts_start)
    print(f"Prompt Similarity: {similarity:.3f}")

    ## Evaluate Responses of prompts
    compare_responses_start = time.time()
    response_similarity = await compare_responses(original_prompt, compressed_prompt)
    print("Compare Response Time: ", time.time() - compare_responses_start)
    print(f"Response Similarity: {response_similarity:.3f}")
    return (execution_time, compressed_ratio, response_similarity, similarity)

async def run_scenario(dataset, target_token):
    avg_execution_time = 0
    avg_response_similarity = 0
    avg_similarity = 0
    avg_compress_ratio = 0
    tasks = []
    batch_size = 20
    for index, data in enumerate(dataset):
        task = asyncio.create_task(analyze_prompt(data, target_token))
        tasks.append(task)
        if (index + 1) % batch_size == 0 or index == len(dataset) - 1:
            results = await asyncio.gather(*tasks)
            for result in results:
                execution_time, compressed_ratio,  response_similarity, similarity = result
                avg_execution_time += execution_time
                avg_response_similarity += response_similarity
                avg_similarity += similarity
                avg_compress_ratio += compressed_ratio
                print(f"Got Batch Size {index}")
            tasks = []
        # Use the index variable as needed
        print(f"Index: {index}")
    
    print(f"Average Execution Time: {avg_execution_time/len(dataset):.3f} seconds")
    print(f"Average Response Similarity: {avg_response_similarity/len(dataset):.3f}")
    print(f"Average Prompt Similarity: {avg_similarity/len(dataset):.3f}")
    print(f"Average Compression Ratio: {avg_compress_ratio/len(dataset):.3f}")


async def main():
    # Load Small Data Set
    small_path = "./examples/nq-open-10_total_documents_gold_at_9.jsonl"
    small_datasets = process_jsonl_file(small_path)

    medium_path = "./examples/nq-open-30_total_documents_gold_at_29.jsonl"
    medium_datasets = process_jsonl_file(medium_path)

    # Run scenarios
    target_token = 500  # Adjust the target token count as needed
    print("Running scenarios for small datasets:")
    await run_scenario(small_datasets[:50], target_token)

    print("Running scenarios for medium datasets:")
    await run_scenario(medium_datasets[:20], target_token)

asyncio.run(main())