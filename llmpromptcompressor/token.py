import tiktoken

openai_encoding = tiktoken.get_encoding("cl100k_base")

def get_token_length(llm:str, prompt:str):
    if( prompt is None or prompt == ""):
        return 0
    
    if( llm == "OPEN_AI"):
        return len(openai_encoding.encode(prompt))
    else:
        raise NotImplementedError("Model not supported")