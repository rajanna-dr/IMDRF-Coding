import os
import json
import numpy as np
from faiss_index import load_annexure_embeddings, build_faiss_index
from faiss_retrieval import query_faiss

def convert_to_serializable(obj):
    """
    Recursively converts all NumPy float32 values in a dictionary or list to Python float.
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.float32):  # Check for NumPy float32
        return float(obj)
    else:
        return obj

def main():
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annexure_folder = os.path.join(project_root, "annexe_json")
    final_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_output.json")
    annex_codes_final_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Annex_codes_final.json")

    # Build Annex A index
    a_file = os.path.join(annexure_folder, "Annex_A_embedded.json")
    codes_a = load_annexure_embeddings(a_file)
    index_a, info_a = build_faiss_index(codes_a, embedding_dim=1536)

    # Build Annex E index
    e_file = os.path.join(annexure_folder, "Annex_E_embedded.json")
    codes_e = load_annexure_embeddings(e_file)
    index_e, info_e = build_faiss_index(codes_e, embedding_dim=1536)

    # Load complaints from final_output.json
    try:
        with open(final_output_path, "r", encoding="utf-8") as f:
            complaints = json.load(f)
    except Exception as e:
        print(f"Error loading complaints file: {e}")
        return

    # Process complaints and retrieve codes
    results = []
    for comp in complaints:
        print("\n=== Complaint ID:", comp.get("complaint_id"), "===\n")
        print("Text:", comp.get("complaint_text"), "\n")

        result_entry = {
            "complaint_id": comp.get("complaint_id"),
            "complaint_text": comp.get("complaint_text"),
            "device_problems": [],
            "patient_problems": []
        }

        # Device problems retrieval
        for term in comp.get("device_problems", []):
            cands = query_faiss(index_a, info_a, term, top_n=6)
            print(f" Device term: '{term}' ⇒ Candidates:")
            device_candidates = []
            for code, desc, dist in cands:
                print(f"   {code} ({dist:.3f}) – {desc}")
                device_candidates.append({"code": code, "description": desc, "distance": dist})
            result_entry["device_problems"].append({"term": term, "candidates": device_candidates})
            print()

        # Patient problems retrieval
        for term in comp.get("patient_problems", []):
            cands = query_faiss(index_e, info_e, term, top_n=6)
            print(f" Patient term: '{term}' ⇒ Candidates:")
            patient_candidates = []
            for code, desc, dist in cands:
                print(f"   {code} ({dist:.3f}) – {desc}")
                patient_candidates.append({"code": code, "description": desc, "distance": dist})
            result_entry["patient_problems"].append({"term": term, "candidates": patient_candidates})
            print()

        results.append(result_entry)

    # Save results to Annex_codes_final.json
    try:
        # Convert results to JSON-serializable format
        serializable_results = convert_to_serializable(results)
        with open(annex_codes_final_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to {annex_codes_final_path}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

if __name__ == "__main__":
    main()