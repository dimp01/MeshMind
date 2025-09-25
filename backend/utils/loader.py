import torch, os
import streamlit as st
from diffusers import StableDiffusionPipeline
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from backend.config import device
from pathlib import Path


SAVE_PATH = Path(__file__).parent.parent / "model" / "diffuser_model"

@st.cache_resource
def load_diffusion_pipeline(device=device):
    """
    Loads and caches the Shap-E pipeline for efficiency.
    Returns: ShapEPipeline object
    """
    if os.path.exists(SAVE_PATH):
        print(f"✅ Model found at: {SAVE_PATH}. Loading pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(
            SAVE_PATH,
            torch_dtype=torch.float16
        ).to(device)
        print("Pipeline loaded successfully! Ready for inference.")
    else:
        print(f"❌ Model not found at: {SAVE_PATH}. Downloading model....")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16
        ).to(device)
        os.makedirs(SAVE_PATH, exist_ok=True)
        pipe.save_pretrained(SAVE_PATH, safe_serialization=True)
        print(f"Model saved for next session to: {SAVE_PATH}")
    return pipe

@st.cache_resource
def get_models(device):
    image_model = load_model("image300M", device=device)
    text_model = load_model("text300M", device=device)
    xm = load_model("transmitter", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))
    return image_model, text_model, xm, diffusion
