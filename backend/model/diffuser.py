import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import decode_latent_mesh
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Shap-E models
image_model = load_model("image300M", device=device)   # Image → 3D model
xm = load_model("transmitter", device=device)
diffusion = diffusion_from_config(load_config("diffusion"))

class DiffusionModel:
    def __init__(self, image_path="temp_img.png"):
      self.image = Image.open(image_path)
      self.image_model = image_model
      self.diffusion = diffusion
      self.xm = xm

    def generate(self):
      batch = [(image_input, None)]
      latents = sample_latents(
          self.image_model,
          self.diffusion,
          batch,
          guidance_scale=3.0,
          progress=True,
          clip_denoised=True,
          use_fp16=True,
          device=device
      )

      return latents

    def save(self, latents, path): 
      mesh = decode_latent_mesh(self.xm, latents[0])
      mesh.export(path)
