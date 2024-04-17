import json
from xopen import xopen
from copy import deepcopy
from tqdm import tqdm

import tiktoken
import time
from llmpromptcompressor import LLMPromptCompressor, settings, RankMethodType
import asyncio

def load_data(path, num_examples=100):
    datasets = []
    with xopen(path) as f:
        for ii, jj in tqdm(enumerate(f), total=num_examples):
            if ii >= num_examples:
                break
            input_example = json.loads(jj)
            question = input_example["question"]
            documents = []
            for ctx in deepcopy(input_example["ctxs"]):
                documents.append(Document.from_dict(ctx))
            prompt = get_qa_prompt(
                question,
                documents,
                mention_random_ordering=False,
                query_aware_contextualization=False,
            )
            c = prompt.split("\n\n")
            instruction, question = c[0], c[-1]
            demonstration = "\n".join(c[1:-1])
            datasets.append({"id": ii, "instruction": instruction, "demonstration": demonstration, "question": question, "answer": input_example["answers"]})
    return datasets


def get_original_prompt(dataset):
    demonstration_str, question, answer = [dataset[key] for key in ["demonstration", "question", "answer"]]
    prompt = "\n\n".join([demonstration_str, question])
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(prompt))
    return (num_tokens, prompt)

async def compress_prompt(dataset, target_token):
    demonstration_str, question, answer = [dataset[key] for key in ["demonstration", "question", "answer"]]

    context_compressor = LLMPromptCompressor( rank_method=RankMethodType.OPEN_AI, concurrent_requests=20, llm_api_config={"open_api_key": settings.OPENAI_API_KEY})
    compressed_prompt = await context_compressor.compress_prompt(
        demonstration_str.split("\n"),
        question=question,
        target_token=target_token,
    )
    return compressed_prompt

def get_token_count(prompt):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(prompt))
    return num_tokens

async def run_scenario(datasets, target_token):
    execution_times = []
    compression_ratios = []
    for i in range(len(datasets)):
        start_time = time.time()
        print("Starting compression for dataset")
        compressed_prompt = await compress_prompt(datasets[i], target_token)
        print("Completed compression for dataset")
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

        origin_tokens = compressed_prompt["origin_tokens"]
        compressed_tokens = compressed_prompt["compressed_tokens"]
        compression_ratio = origin_tokens / compressed_tokens
        compression_ratios.append(compression_ratio)

    avg_execution_time = sum(execution_times) / len(execution_times)
    avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)

    #print(f"Scenario: Context Filtering={context_filtering}, Sentence Filtering={sentence_filtering}, Token Level Filtering={token_level_filtering}")
    print(f"Average Execution Time: {avg_execution_time:.3f} seconds for original tokens {origin_tokens}, Compressed Tokens {compressed_tokens} ")
    print(f"Average Compression Ratio: {avg_compression_ratio:.2f}")
    print()


def process_jsonl_file(file_path):
    data = []

    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line)
            
            question = record['question']
            answer = record['answers'][0]  # Assuming only one answer per question
            
            demonstration = ''
            for ctx in record['ctxs']:
                title = ctx['title']
                text = ctx['text']
                demonstration += f"Title: {title} : {text}\n\n"
            
            data.append({
                'question': question,
                'answer': answer,
                'demonstration': demonstration.strip()
            })
    
    return data


async def main():
    # Load Small Data Set
    small_path = "./examples/nq-open-10_total_documents_gold_at_9.jsonl"
    small_datasets = process_jsonl_file(small_path)

    # Run scenarios
    target_token = 500  # Adjust the target token count as needed
    print("Running scenarios for small datasets:")
    await run_scenario(small_datasets[:2], target_token)

asyncio.run(main())