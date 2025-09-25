from datetime import datetime
from backend.config import genai
import hashlib

def text_model_prompt(product_name, dimensions, features, materials, style, intended_use, colors):
    """
    Generates a Shap-E compatible prompt using Gemini.
    """
    if isinstance(features, (list, tuple)):
        features_text = ", ".join(features)
    else:
        features_text = features

    message = f"""
    Make a 3D model of a {product_name}, color: {colors}, 
    features: [{features_text}], material: {materials},
    style: {style}, purpose: {intended_use}, dimensions: {dimensions}
    """

    prompt = f"""
    You are a prompt expert.
    Generate ONLY the prompt for Shap-E by improving this prompt.
    \"{message}\"
    Don't give any text except the prompt.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt.strip())
        return response.text.strip()
    except Exception as e:
        return f"An error occurred in Gemini: {e}"

def diffusion_model_prompt(product_name, dimensions, style, colors, features):
    """
    Generates a Shap-E compatible prompt using Gemini.
    """
    if isinstance(features, (list, tuple)):
        features_text = ", ".join(features)
    else:
        features_text = features

    prompt = f"""
    You are a prompt expert.
    Generate ONLY the prompt for Image generation of a product by using this info.

    - Model: Runaway
    - Product: {product_name}
    - Dimension: {dimensions}
    - Style/Design: {style}
    - Colors: {colors}
    - Important features: {features_text}

    The image should be photorealistic, clean, and symmetrical, suitable for presentation or marketing.
    Avoid cartoonish, low-resolution, or distorted elements.
    Focus entirely on the object with accurate details.
    Don't give any text except the prompt.
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt.strip())
        return response.text.strip()
    except Exception as e:
        return f"An error occurred in Gemini: {e}"

def gen_file_name(prompt: str) -> str:
    """
    Generates a short, safe filename for saving 3D objects.
    """
    try:
        response = model.generate_content(f"Give only a 2 word short filename for (no other text just 2 word reponse): {prompt}")
        safe_prompt = "".join(c for c in response.text if c.isalnum() or c in " _-").rstrip()
        # Add hash to avoid duplicates
        short_hash = hashlib.md5(prompt.encode()).hexdigest()[:6]
        filename = f"{safe_prompt}_{short_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
        return filename
    except Exception as e:
        return f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
