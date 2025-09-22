from backend.model.text import load_shap_e_pipeline
from backend.model.loader import gen_image, load_diffusion_pipeline
from backend.model.diffuser import DiffusionModel
from backend.config import device


text_pipe = load_shap_e_pipeline(device)
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
        decoder_output = text_pipe(
            self.prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.steps,
            frame_size=self.frame_size,
            output_type=self.output_type,
            return_dict=self.return_dict,
        )

        return decoder_output

    def diffusion(self):
        image = gen_image(self.prompt, diffusion_p)
        diffuser = DiffusionModel(image)
        latents = diffuser.generate(self.guidance_scale)
        return latents

