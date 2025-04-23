import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration parameters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-default-api-key")
ANNEXURE_FOLDER = os.path.join("annexe_json")
COMPLAINTS_FILE = os.path.join("data", "complaints.xlsx")