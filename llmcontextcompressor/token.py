import tiktoken
from nltk.tokenize import word_tokenize

from llmcontextcompressor.constants import RankMethodType

openai_encoding = tiktoken.get_encoding("cl100k_base")

def get_token_length(llm_type: RankMethodType, prompt:str):
    if( prompt is None or prompt == ""):
        return 0
    
    if( llm_type == RankMethodType.OPEN_AI):
        return len(openai_encoding.encode(prompt))
    elif( llm_type == RankMethodType.BM25):
        return len(word_tokenize(prompt))
    else:
        raise NotImplementedError("Model not supported")