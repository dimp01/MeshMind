import torch
import streamlit as st
from diffusers import StableDiffusionPipeline
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from backend.config import device


@st.cache_resource
def load_diffusion_pipeline(device=device):
    """
    Loads and caches the Shap-E pipeline for efficiency.
    Returns: ShapEPipeline object
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)
    return pipe

@st.cache_resource
def get_models(device):
    image_model = load_model("image300M", device=device)
    text_model = load_model("text300M", device=device)
    xm = load_model("transmitter", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))
    return image_model, text_model, xm, diffusion
