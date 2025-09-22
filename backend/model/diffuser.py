from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import decode_latent_mesh
from backend.config import device
from io import BytesIO
from PIL import Image
import gc

# Load Shap-E models globally to save time
image_model = load_model("image300M", device=device)  # Image → 3D
xm = load_model("transmitter", device=device)
diffusion = diffusion_from_config(load_config("diffusion"))


class DiffusionModel:
    def __init__(self, image_bytes):
        # Convert bytes to PIL Image
        self.image = Image.open(BytesIO(image_bytes.getvalue())).convert("RGB")
        self.image_model = image_model
        self.diffusion = diffusion
        self.xm = xm

        # Default generation parameters
        self.guidance_scale = 3.0
        self.use_karras = True
        self.karras_steps = 64
        self.sigma_min = 1e-3
        self.sigma_max = 160
        self.s_churn = 0
        self.clip_denoised = True
        self.use_fp16 = True
        self.progress = True

    def generate(
        self,
        guidance_scale=None,
        karras_steps=None,
        sigma_min=None,
        sigma_max=None,
        s_churn=None,
        use_karras=None,
        clip_denoised=None,
        use_fp16=None,
        progress=None,
    ):
        # Update parameters if provided
        guidance_scale = guidance_scale or self.guidance_scale
        karras_steps = karras_steps or self.karras_steps
        sigma_min = sigma_min or self.sigma_min
        sigma_max = sigma_max or self.sigma_max
        s_churn = s_churn or self.s_churn
        use_karras = use_karras if use_karras is not None else self.use_karras
        clip_denoised = clip_denoised if clip_denoised is not None else self.clip_denoised
        use_fp16 = use_fp16 if use_fp16 is not None else self.use_fp16
        progress = progress if progress is not None else self.progress

        # Prepare input for Shap-E
        model_kwargs = {"images": [self.image]}

        # Generate latent 3D representation
        latents = sample_latents(
            batch_size=1,
            model=self.image_model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=model_kwargs,
            progress=progress,
            clip_denoised=clip_denoised,
            use_fp16=use_fp16,
            device=device,
            use_karras=use_karras,
            karras_steps=karras_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            s_churn=s_churn,
        )

        mesh = decode_latent_mesh(self.xm, latents[0])
        return mesh
