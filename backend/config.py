import os
import torch
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("⚠️ GEMINI_API_KEY not found in .env file.")
os.environ["GEMINI_API_KEY"] = api_key
genai.configure(api_key=api_key)

# Expose objects
__all__ = ["device", "genai"]
