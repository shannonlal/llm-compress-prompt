from llmpromptcompressor import LLMPromptCompressor, settings, RankMethodType
import json
import tiktoken

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