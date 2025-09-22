import torch
from diffusers import StableDiffusionPipeline
from rembg import remove
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)

def gen_image(prompt, file_path="temp_img.png"):
    image = pipe(prompt, guidance_scale=7.5).images[0]
    image_no_bg = remove(image)
    image_no_bg.save(file_path)
