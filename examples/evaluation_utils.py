
import numpy as np
from openai import AsyncOpenAI
from llm_utils import get_gpt_35_response


async def compare_prompts(original_prompt, compressed_prompt):
    client = AsyncOpenAI()
    # Get the embeddings for the original and compressed prompts
    original_embedding_promise = await client.embeddings.create(
      input=original_prompt,
      model="text-embedding-3-small"
    )
    original_embedding = original_embedding_promise.data[0].embedding

    compressed_embedding_promise = await client.embeddings.create(
      input=compressed_prompt,
      model="text-embedding-3-small"
    )
    compressed_embedding = compressed_embedding_promise.data[0].embedding

    # Convert the embeddings to numpy arrays
    original_embedding = np.array(original_embedding)
    compressed_embedding = np.array(compressed_embedding)

    # Calculate the cosine similarity between the embeddings
    similarity = np.dot(original_embedding, compressed_embedding) / (np.linalg.norm(original_embedding) * np.linalg.norm(compressed_embedding))

    return similarity


async def compare_responses(original_prompt, compressed_prompt):
    client = AsyncOpenAI()
    # Get the responses from GPT-4 for the original and compressed prompts
    original_response = await get_gpt_35_response(original_prompt)
    compressed_response = await get_gpt_35_response(compressed_prompt)

    # Get the embeddings for the responses
    original_response_embedding_promise = await client.embeddings.create(
      input=original_response,
      model="text-embedding-3-small"
    )

    original_response_embedding = original_response_embedding_promise.data[0].embedding
    compressed_response_embedding_promise = await client.embeddings.create(
      input=compressed_response,
      model="text-embedding-3-small"
    )
    compressed_response_embedding = compressed_response_embedding_promise.data[0].embedding

    # Convert the embeddings to numpy arrays
    original_response_embedding = np.array(original_response_embedding)
    compressed_response_embedding = np.array(compressed_response_embedding)

    # Calculate the cosine similarity between the response embeddings
    similarity = np.dot(original_response_embedding, compressed_response_embedding) / (np.linalg.norm(original_response_embedding) * np.linalg.norm(compressed_response_embedding))

    return similarity