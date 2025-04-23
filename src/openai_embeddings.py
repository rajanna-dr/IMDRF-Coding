# openai_embeddings.py

import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_embedding(text):
    """
    Compute an embedding for the given text using the OpenAI API.
    
    Parameters:
      text (str): The text for which to compute the embedding.
      
    Returns:
      list: A list of floats representing the embedding vector.
    """
    try:
        # Call the OpenAI embedding API
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"  # Use the recommended embedding model
        )
        # The response returns a list in the 'data' key. Each item has an 'embedding' field.
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        print("An error occurred while obtaining the embedding:", e)
        return None

if __name__ == "__main__":
    # Sample text
    sample_text = "This is a sample sentence to compute an embedding."
    
    # Get the embedding
    embedding_vector = get_openai_embedding(sample_text)
    
    # Print the resulting embedding vector
    if embedding_vector is not None:
        print("Embedding for the sample text:")
        print(embedding_vector)
    else:
        print("Failed to compute embedding.")
