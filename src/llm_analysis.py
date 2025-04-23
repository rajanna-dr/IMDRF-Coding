# src/llm_analysis.py

import os
import json
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment.
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_complaint(complaint_text):
    """
    Uses the OpenAI API to analyze the complaint text and extract device and patient problems.
    
    The prompt now provides explicit instructions and examples, allowing for longer outputs.
    
    Expected JSON format:
    {
      "complaint_text": "<original complaint text>",
      "device_problems": ["<list of phrases describing the device issue>"],
      "patient_problems": ["<list of phrases describing the patient issue or clinical signs>"]
    }
    
    If no problems are identified in a category, return an empty list for that category.
    
    Parameters:
      complaint_text (str): The complaint text to analyze.
      
    Returns:
      dict: A dictionary containing the extracted data. In case of error, returns the original text with empty lists.
    """
    # Enhanced prompt with explicit instructions and an example.
    prompt = f"""
As a medical device expert, analyze the complaint below and extract:
1. **Device Problems**: Identify phrases indicating device malfunctions/issues. Focus on these categories:
   - Patient-Device Interaction (e.g., rejection, anatomical mismatch)
   - Manufacturing/Packaging/Shipping (e.g., damaged packaging, missing components)
   - Chemical (e.g., unusual odor, particulates)
   - Material Integrity (e.g., fractures, corrosion)
   - Mechanical (e.g., deployment failure, leaks)
   - Optical (e.g., visual distortion)
   - Electrical/Electronic (e.g., power failure, battery issues)

2. **Patient Problems**: Extract symptoms/adverse effects experienced by the patient(s). The patient problems may be or may not be because of device use. But if any problems faced by the patient is mentioned, extract them.
    Please note that if a body part is mentioned or an organ system is mentioned, it has to be extracted in full (eg. eye pain, pain in limb, reduced secretion in throat, swelling of the finger etc.)
Return JSON format with:
{{
  "complaint_text": "original text",
  "device_problems": ["extracted_phrases"],
  "patient_problems": ["extracted_symptoms"]
}}

Example for "Catheter leaked during use causing urinary tract infection":
{{
  "complaint_text": "Catheter leaked during use causing infection",
  "device_problems": ["Catheter leaked"],
  "patient_problems": ["urinary tract infection"]
}}

Now analyze:
"{complaint_text}"
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini-2024-07-18",  # You may adjust the model if needed.
            messages=[
                {"role": "system", "content": "You are an expert medical complaint analyzer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8192,   # Increased token limit to accommodate longer inputs/outputs.
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        output_text = response['choices'][0]['message']['content'].strip()
        # Parse the output as JSON.
        result = json.loads(output_text)
        # Ensure keys exist.
        if "device_problems" not in result:
            result["device_problems"] = []
        if "patient_problems" not in result:
            result["patient_problems"] = []
        return result
    except Exception as e:
        print("Error during LLM analysis:", e)
        return {
            "complaint_text": complaint_text,
            "device_problems": [],
            "patient_problems": []
        }

if __name__ == "__main__":
    # Test with a sample complaint text.
    sample_text = (
        "Event Description: During the procedure, the stent failed to deploy properly and malfunctioned, "
        "resulting in the patient experiencing severe pain, dizziness, and a feeling of lightheadedness. "
        "There were multiple technical issues observed with the device."
    )
    analysis = analyze_complaint(sample_text)
    
    print("LLM Analysis Result:")
    print(json.dumps(analysis, indent=2))
