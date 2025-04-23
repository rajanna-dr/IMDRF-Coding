# test_embedding.py

import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        print("Error computing embedding for text:", text, "\nError:", e)
        return None

if __name__ == "__main__":
    sample_text = "This is a test description for a device code."
    emb = get_openai_embedding(sample_text)
    if emb is not None:
        print("Embedding computed successfully:")
        print(emb)
    else:
        print("Failed to compute embedding.")
