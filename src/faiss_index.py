# src/faiss_index.py

import os
import json
import numpy as np
import faiss

def load_annexure_embeddings(annexure_file):
    """
    Loads precomputed embeddings from an annexure JSON file.
    Supports either a dictionary with a "codes" key or a list of code dictionaries.
    Each code dictionary must have a non-empty "embedding" field.
    """
    with open(annexure_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "codes" in data:
        codes = data["codes"]
    else:
        codes = data
    # Filter out entries with missing or empty embeddings.
    codes = [code for code in codes if code.get("embedding") and len(code.get("embedding")) > 0]
    return codes

def build_faiss_index(codes, embedding_dim=1536):
    """
    Builds a FAISS index from a list of code dictionaries.
    
    Parameters:
      codes (list): List of codes with the "embedding" key.
      embedding_dim (int): The dimension of the embeddings.
    
    Returns:
      index: A FAISS index containing all embeddings.
      code_info: A list of dictionaries mapping index positions to code details.
    """
    embeddings = np.array([code["embedding"] for code in codes]).astype("float32")
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # Verify that the embeddings have the correct dimensionality.
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(f"Expected embedding dimension {embedding_dim}, but got {embeddings.shape[1]}")
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    
    code_info = [{"code": code["code"], "description": code.get("description", "No description available")} for code in codes]
    
    return index, code_info

if __name__ == "__main__":
    annexure_folder = "annexe_json"
    
    # Process Annex A
    annex_a_file = os.path.join(annexure_folder, "Annex_A_embedded.json")
    codes_a = load_annexure_embeddings(annex_a_file)
    print(f"Loaded {len(codes_a)} codes from Annex A.")
    index_a, code_info_a = build_faiss_index(codes_a, embedding_dim=1536)
    print("FAISS index for Annex A built with", index_a.ntotal, "codes.")
    
    # Process Annex E
    annex_e_file = os.path.join(annexure_folder, "Annex_E_embedded.json")
    codes_e = load_annexure_embeddings(annex_e_file)
    print(f"Loaded {len(codes_e)} codes from Annex E.")
    index_e, code_info_e = build_faiss_index(codes_e, embedding_dim=1536)
    print("FAISS index for Annex E built with", index_e.ntotal, "codes.")
