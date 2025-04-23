# src/embed_annexures.py

import os
import json
import openai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_embedding(text):
    """
    Computes the embedding for the provided text using OpenAI's text-embedding-ada-002 model.
    
    Parameters:
      text (str): The text to embed.
      
    Returns:
      list: A list of floats representing the embedding vector, or None if an error occurs.
    """
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

def embed_annexure(file_path, output_path):
    """
    Reads an annexure JSON file, computes embeddings for each code entry by concatenating the 'term'
    and 'definition' fields (if available), adds the embedding to each code entry, and writes the
    updated JSON to a new file.
    
    The input JSON can be either:
      - A dictionary with a "codes" key: {"codes": [{...}, {...}, ...]}
      - A list of code dictionaries: [{...}, {...}, ...]
    
    Each code dictionary should have at least:
      - "code": the code identifier,
      - "term": the short term.
      - "definition": the detailed definition of the code.
    
    Parameters:
      file_path (str): Path to the input annexure JSON file.
      output_path (str): Path to write the updated JSON file with embeddings.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Determine whether data is a dict with key "codes" or a list.
    if isinstance(data, dict) and "codes" in data:
        codes = data.get("codes", [])
    elif isinstance(data, list):
        codes = data
    else:
        print(f"Unexpected data format in {file_path}.")
        return

    # Process each code entry.
    for item in codes:
        term = item.get("term", "")
        definition = item.get("definition", "")
        # Concatenate term and definition.
        # Only include fields that are not empty.
        text_parts = []
        if term:
            text_parts.append(term)
        if definition:
            text_parts.append(definition)
        # Optionally, you can also include the code in the embedding text if needed.
        # For example: text_parts.append(f"Code: {item.get('code', '')}")
        embedding_text = " ".join(text_parts)
        
        if embedding_text:
            embedding = get_openai_embedding(embedding_text)
            item["embedding"] = embedding
        else:
            item["embedding"] = None

    # Update data with the processed codes.
    if isinstance(data, dict) and "codes" in data:
        data["codes"] = codes
    else:
        data = codes

    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, indent=2)
        print(f"Embedded annexure saved to {output_path}")
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")

def main():
    # Set the folder where your annexure files are stored.
    annexure_folder = "annexe_json"
    annex_a_file = os.path.join(annexure_folder, "Annex_A.json")
    annex_e_file = os.path.join(annexure_folder, "Annex_E.json")

    # Define output file paths for the embedded versions.
    annex_a_embedded = os.path.join(annexure_folder, "Annex_A_embedded.json")
    annex_e_embedded = os.path.join(annexure_folder, "Annex_E_embedded.json")

    # Embed both Annex A and Annex E.
    embed_annexure(annex_a_file, annex_a_embedded)
    embed_annexure(annex_e_file, annex_e_embedded)

if __name__ == "__main__":
    main()
for idx, item in enumerate(codes):
    term = item.get("term", "")
    definition = item.get("definition", "")
    text_parts = []
    if term:
        text_parts.append(term)
    if definition:
        text_parts.append(definition)
    embedding_text = " ".join(text_parts)
    
    if embedding_text:
        embedding = get_openai_embedding(embedding_text)
        item["embedding"] = embedding
        print(f"Processed code {item.get('code')} ({idx + 1} of {len(codes)})")
    else:
        item["embedding"] = None
