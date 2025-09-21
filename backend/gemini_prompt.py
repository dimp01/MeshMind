from datetime import datetime
from backend.config import genai
import hashlib

model = genai.GenerativeModel("gemini-1.5-flash")

def generate_shap_e_prompt(product_type, dimensions, features, materials, style, intended_use):
    """
    Generates a Shap-E compatible prompt using Gemini.
    """
    if isinstance(features, (list, tuple)):
        features_text = ", ".join(features)
    else:
        features_text = features

    prompt = f"""
    You are an expert product designer.
    Generate a prompt for Shap-E to create a 3D model of a {product_type} with these specs:

    - Dimensions: {dimensions}
    - Core Features: {features_text}
    - Materials/Finish: {materials}
    - Style/Design: {style}
    - Intended Use: {intended_use}

    The design should be realistic, clean, symmetrical, and manufacturable.
    Avoid cartoonish, distorted, or unrealistic elements.
    """
    try:
        response = model.generate_content(prompt.strip())
        return response.text.strip()
    except Exception as e:
        return f"An error occurred in Gemini: {e}"

def gen_file_name(prompt: str) -> str:
    """
    Generates a short, safe filename for saving 3D objects.
    """
    try:
        response = model.generate_content(f"Give a 2 word short filename for: {prompt}")
        safe_prompt = "".join(c for c in response.text if c.isalnum() or c in " _-").rstrip()
        # Add hash to avoid duplicates
        short_hash = hashlib.md5(prompt.encode()).hexdigest()[:6]
        filename = f"{safe_prompt}_{short_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
        return filename
    except Exception as e:
        return f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
