import openai
import os
import numpy as np  # Import NumPy for array handling

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_embedding(text):
    """
    Computes the embedding for the provided text using OpenAI's text-embedding-ada-002 model.
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error computing embedding for text '{text}': {e}")
        return None

def query_faiss(index, info, query, top_n=6):
    """
    Queries the FAISS index with the given query text and returns the top N results.
    """
    embedding = get_openai_embedding(query)
    if embedding is None:
        return []

    # Convert the embedding to a NumPy array
    embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)

    distances, indices = index.search(embedding, top_n)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        code = info[idx]["code"]
        description = info[idx]["description"]
        results.append((code, description, dist))
    return results