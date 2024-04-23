import os
import anthropic
from openai import AsyncOpenAI

ANTHROPIC_API_KEY =  os.environ["ANTHROPIC_API_KEY"]

def get_anthropic_response(prompt):
    client = anthropic.Anthropic(
      api_key=ANTHROPIC_API_KEY,
    )

    response = client.messages.create(
      model="claude-3-opus-20240229",
      max_tokens=3785,
      temperature=0,
      system="You are a smart digital assistant",
      messages=[{
          "role": "user",
          "content": [
              {
                  "type": "text",
                  "text": prompt
              }
          ]}
      ])


    return response.content[0].text

async def get_gpt_35_response(prompt):
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

async def get_openai_embedding(prompt):
    client = AsyncOpenAI()
    # Get the embeddings for the responses
    embedding_promise = await client.embeddings.create(
      input=prompt,
      model="text-embedding-3-small"
    )
    return embedding_promise.data[0].embedding