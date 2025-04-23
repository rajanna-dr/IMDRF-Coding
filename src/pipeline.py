#!/usr/bin/env python3
# run_pipeline.py

import subprocess
import json
import os
from dotenv import load_dotenv

# 1) Run your ingestion step (which writes src/final_output.json)
subprocess.run(["python", "src/ingestion.py"], check=True)

# 2) Now load final_output.json
final_out_path = os.path.join("src", "final_output.json")
with open(final_out_path, "r", encoding="utf-8") as f:
    complaints = json.load(f)

# 3) Build (or reload) your FAISS indexes
from src.faiss_index import load_annexure_embeddings, build_faiss_index
from src.faiss_retrieval import query_faiss

load_dotenv()  # if you need embeddings or OpenAI key loaded here

ann_folder = "annexe_json"
# Annex A (device)
a_file = os.path.join(ann_folder, "Annex_A_embedded.json")
codes_a = load_annexure_embeddings(a_file)
index_a, info_a = build_faiss_index(codes_a, embedding_dim=1536)

# Annex E (patient)
e_file = os.path.join(ann_folder, "Annex_E_embedded.json")
codes_e = load_annexure_embeddings(e_file)
index_e, info_e = build_faiss_index(codes_e, embedding_dim=1536)

# 4) Loop through complaints, retrieve codes, assemble results
results = []
for comp in complaints:
    rec = {
        "complaint_id": comp.get("complaint_id"),
        "complaint_text": comp.get("complaint_text"),
        "device_problems": comp.get("device_problems", []),
        "patient_problems": comp.get("patient_problems", []),
        "mapped_device_codes": [],
        "mapped_patient_codes": []
    }

    # device
    seen = set()
    for term in rec["device_problems"]:
        for code, dist in query_faiss(index_a, info_a, term, top_n=6):
            if code not in seen:
                seen.add(code)
                rec["mapped_device_codes"].append(code)

    # patient
    seen = set()
    for term in rec["patient_problems"]:
        for code, dist in query_faiss(index_e, info_e, term, top_n=6):
            if code not in seen:
                seen.add(code)
                rec["mapped_patient_codes"].append(code)

    results.append(rec)

# 5) Write out your second JSON
out2 = os.path.join("src", "final_code_output.json")
with open(out2, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("✅ Generated two files:")
print("  • src/final_output.json")
print("  • src/final_code_output.json")
