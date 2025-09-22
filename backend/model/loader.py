import torch
from diffusers import StableDiffusionPipeline
from backend.config import device
from rembg import remove
from io import BytesIO
from PIL import Image
import streamlit as st


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

def gen_image(prompt, pipe):
    image_bytes = BytesIO()
    image = pipe(prompt, guidance_scale=7.5).images[0]
    image_no_bg = remove(image)
    image_no_bg.save(image_bytes)
    return image_bytes
