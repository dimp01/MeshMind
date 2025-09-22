import torch
import streamlit as st
from diffusers import ShapEPipeline
from backend.config import device

@st.cache_resource
def load_shap_e_pipeline(device=device):
    """
    Loads and caches the Shap-E pipeline for efficiency.
    Returns: ShapEPipeline object
    """
    pipe = ShapEPipeline.from_pretrained(
        "openai/shap-e",
        torch_dtype=torch.float16
    ).to(device)
    return pipe


