
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config

from backend.model.loader import gen_image, load_diffusion_pipeline
from backend.model.text import TextModel
from backend.model.diffuser import DiffusionModel
from backend.config import device

# Load Shap-E models globally to save time
image_model = load_model("image300M", device=device)
text_model = load_model('text300M', device=device)
xm = load_model("transmitter", device=device)
diffusion = diffusion_from_config(load_config("diffusion"))

diffusion_p = load_diffusion_pipeline(device)

class GenerateModel:
    def __init__(
        self,
        prompt,
        guidance_scale,
        num_inference_steps,
        frame_size,
        output_type,
        return_dict,
    ):
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.steps = num_inference_steps
        self.frame_size = frame_size
        self.output_type = output_type
        self.return_dict = return_dict

    def text(self):
        text = TextModel(text_model, diffusion, xm)
        latents = text.generate(
            self.prompt,
            guidance_scale=self.guidance_scale,
            karras_steps=self.steps,
        )

        return lantents

    def diffusion(self):
        image = gen_image(self.prompt, diffusion_p)
        diffuser = DiffusionModel(image, image_model, diffusion, xm)
        latents = diffuser.generate(self.guidance_scale)
        return latents

