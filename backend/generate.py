from backend.utils.loader import get_models, load_diffusion_pipeline
from backend.utils.text import TextModel
from backend.utils.diffuser import DiffusionModel
from backend.config import device
import torch

image_model, text_model, xm, diffusion = get_models(device)
# diffusion_p = load_diffusion_pipeline(device)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
            sigma_max=self.frame_size
        )

        return latents

    def diffusion(self):
        diffuser = DiffusionModel(image_model, diffusion, xm)
        image = diffuser.gen_image(self.prompt, diffusion_p)
        latents = diffuser.generate(
            guidance_scale=self.guidance_scale,
            sigma_max=self.frame_size,
            karras_steps=self.steps
        )
        return latents

