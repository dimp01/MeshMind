from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import decode_latent_mesh
from backend.config import device
from rembg import remove
from io import BytesIO
from PIL import Image


class DiffusionModel:
    def __init__(self, image_model, diffusion, xm):
        # Convert bytes to PIL Image
        # self.image = Image.open(BytesIO(self.image_bytes.getvalue()))
        self.image_model = image_model
        self.diffusion = diffusion
        self.xm = xm

        # Default generation parameters
        self.guidance_scale = 3.5
        self.use_karras = True
        self.karras_steps = 160
        self.sigma_min = 1e-3
        self.sigma_max = 80
        self.s_churn = 0
        self.clip_denoised = True
        self.use_fp16 = False
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
        
    def gen_image(self, prompt, pipe):
        self.image = BytesIO()
        image_s = pipe(prompt, guidance_scale=7.5).images[0]
        image_no_bg = remove(image_s)
        image_no_bg.save(self.image, format="PNG")

