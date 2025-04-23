# src/ingestion.py

import os
import sys
import json
import pandas as pd

# For Word documents
try:
    from docx import Document
except ImportError:
    Document = None

# Import the LLM analysis function.
from llm_analysis import analyze_complaint

# Candidate column names for complaint text and complaint/report id.
COMPLAINT_TEXT_CANDIDATES = ["complaint_text", "event text", "EVENT TEXT", "Event Text", "event_text"]
COMPLAINT_ID_CANDIDATES = ["complaint_number", "report_number", "id", "identifier"]

def find_column(columns, candidate_names):
    """
    Searches for a column in a list of column names that matches one of the candidate names (case-insensitive).
    """
    lower_columns = {col.lower(): col for col in columns}
    for candidate in candidate_names:
        if candidate.lower() in lower_columns:
            return lower_columns[candidate.lower()]
    return None

def ingest_complaints(file_path):
    """
    Reads complaint data from a file. Supported formats:
      - Excel (.xlsx, .xls) / CSV (.csv): Searches for candidate columns for complaint text and ID.
      - Text (.txt): Each non-empty line is a complaint.
      - Word (.docx): Each non-empty paragraph is a complaint.
      - JSON (.json): Expects a list of complaints or a dictionary with a "complaints" key.
    
    Returns:
      list: A list of dictionaries, each with keys "complaint_id" and "complaint_text".
    """
    ext = os.path.splitext(file_path)[1].lower()
    complaints = []

    if ext in ['.xlsx', '.xls', '.csv']:
        try:
            if ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")

        text_col = find_column(df.columns, COMPLAINT_TEXT_CANDIDATES)
        if not text_col:
            raise ValueError(f"The file must contain one of these columns for complaint text: {COMPLAINT_TEXT_CANDIDATES}")
        id_col = find_column(df.columns, COMPLAINT_ID_CANDIDATES)

        for idx, row in df.iterrows():
            text = row.get(text_col)
            if pd.isna(text) or not str(text).strip():
                continue
            complaint_id = row.get(id_col) if id_col and not pd.isna(row.get(id_col)) else None
            complaints.append({
                "complaint_id": complaint_id,
                "complaint_text": str(text).strip()
            })

    elif ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            for idx, line in enumerate(lines, start=1):
                complaints.append({
                    "complaint_id": idx,
                    "complaint_text": line
                })
        except Exception as e:
            raise Exception(f"Error reading text file: {e}")

    elif ext == '.docx':
        if Document is None:
            raise ImportError("python-docx is not installed. Please install it.")
        try:
            document = Document(file_path)
            paragraphs = [para.text.strip() for para in document.paragraphs if para.text.strip()]
            for idx, text in enumerate(paragraphs, start=1):
                complaints.append({
                    "complaint_id": idx,
                    "complaint_text": text
                })
        except Exception as e:
            raise Exception(f"Error reading Word document: {e}")

    elif ext == '.json':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and "complaints" in data:
                data = data["complaints"]
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of complaints or have a 'complaints' key.")
            for idx, item in enumerate(data, start=1):
                if isinstance(item, dict):
                    text_col = find_column(item.keys(), COMPLAINT_TEXT_CANDIDATES)
                    if not text_col:
                        continue  # Skip if no text field is found.
                    text = str(item.get(text_col, "")).strip()
                    complaint_id = item.get(find_column(item.keys(), COMPLAINT_ID_CANDIDATES), None)
                    if text:
                        complaints.append({
                            "complaint_id": complaint_id if complaint_id is not None else idx,
                            "complaint_text": text
                        })
                elif isinstance(item, str):
                    text = item.strip()
                    if text:
                        complaints.append({
                            "complaint_id": idx,
                            "complaint_text": text
                        })
        except Exception as e:
            raise Exception(f"Error reading JSON file: {e}")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return complaints

def process_complaints(file_path):
    """
    Ingests complaints from a file, processes each using the LLM analysis module, and returns the results.
    
    Each complaint is analyzed by sending its text to LLM analysis to extract device and patient problems.
    
    Returns:
      list: A list of dictionaries with keys:
            - "complaint_id"
            - "complaint_text"
            - "device_problems" (list)
            - "patient_problems" (list)
    """
    complaints = ingest_complaints(file_path)
    processed_results = []

    for comp in complaints:
        analysis_result = analyze_complaint(comp["complaint_text"])
        analysis_result["complaint_id"] = comp["complaint_id"]
        processed_results.append(analysis_result)

    return processed_results

if __name__ == "__main__":
    # Use the provided file or default to "data/complaints.xlsx"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = os.path.join("data", "complaints.xlsx")

    results = process_complaints(file_path)

    # Print the JSON output to the terminal.
    print(json.dumps(results, indent=2))

    # Determine a file path to write the JSON output.
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_filepath = os.path.join(project_root, "final_output.json")

    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Failed to write output to {output_filepath}: {e}")
