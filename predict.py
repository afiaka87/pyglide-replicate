import torch as th
import cog
import util
import pathlib

import sys
sys.path.append("glide-text2im")
from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint

class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Initialize the model to cache it
        self.base_model, self.base_diffusion, self.base_options = util.load_glide(
            model_name="base"
        )
        self.sr_model, self.sr_diffusion, self.sr_options = util.load_glide(
            model_name="upsample"
        )
        self.clip_model = create_clip_model(device="cuda")
        self.clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', "cuda"))
        self.clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', "cuda"))

    @cog.input(
        "prompt",
        type=str,
        default="",
        help="Text prompt to use. Keep it simple/literal and avoid using poetic language (unlike CLIP).",
    )
    @cog.input(
        "batch_size",
        type=int,
        default=4,
        help="Batch size. Number of generations to predict",
        min=1,
        max=4,
    )
    @cog.input(
        "side_x",
        type=int,
        default=64,
        help="Must be multiple of 8. Going above 64 is not recommended. Actual image will be 4x larger. Using 64 enables guidance with released noisy CLIP",
        options=[32, 48, 64, 80, 96, 112, 128]
    )
    @cog.input(
        "side_y", 
        type=int,
        default=64,
        help="Must be multiple of 8. Going above 64 is not recommended. Actual image size will be 4x larger. Using 64 enables guidance with released noisy CLIP",
        options=[32, 48, 64, 80, 96, 112, 128]
    )
    @cog.input(
        "guidance_scale",
        type=float,
        default=4,
        help="Classifier-free guidance scale. Higher values move further away from unconditional outputs. Lower values move closer to unconditional outputs. Negative values guide towards semantically opposite classes. 4-16 is a reasonable range.",
    )
    @cog.input(
        "upsample_temp",
        type=float,
        default=0.997,
        help="Upsample temperature. Consider lowering to ~0.997 for blurry images with fewer artifacts.",
        min=0.75,
        max=1.0,
    )
    @cog.input(
        "timestep_respacing",
        type=str,
        default="50",
        help="Number of timesteps to use for base model. Going above 150 has diminishing returns.",
        options=["25", "50", "100", "150", "250"],
    )
    @cog.input(
        "seed",
        type=int,
        default=0,
        help="Seed for reproducibility",
    )
    def predict(
        self,
        prompt,
        batch_size,
        side_x,
        side_y,
        guidance_scale,
        upsample_temp,
        timestep_respacing,
        seed,
    ):
        th.manual_seed(seed)
        # Run this again to change the model parameters
        self.base_model, self.base_diffusion, self.base_options = util.load_glide(
            model_name="base", timestep_respacing=timestep_respacing
        )
        self.base_model.to("cuda")
        self.sr_model.to("cuda")
        with th.no_grad():
            # Setup guidance function for CLIP model.
            clip_cond_fn = None
            if side_x == 64 and side_y == 64:
                clip_cond_fn = self.clip_model.cond_fn([prompt] * 2 * batch_size, guidance_scale)
            base_samples = util.sample(
                self.base_model,
                self.base_diffusion,
                self.base_options,
                side_x=side_x,
                side_y=side_y,
                prompt=prompt,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
                device="cuda",
                clip_cond_fn=clip_cond_fn
            )
            base_pil_images = util.pred_to_pil(base_samples)
            base_pil_images.save("/src/base_predictions.png")
            yield pathlib.Path("/src/base_predictions.png")

            sr_samples = util.sample_sr(
                self.sr_model,
                self.sr_diffusion,
                self.sr_options,
                samples=base_samples,
                prompt=prompt,
                sr_x=int(side_x*4),
                sr_y=int(side_y*4),
                upsample_temp=upsample_temp,
                batch_size=batch_size,
                device="cuda",
            )
            sr_pil_images = util.pred_to_pil(sr_samples)
            sr_pil_images.save("/src/upsample_predictions.png")
            yield pathlib.Path("/src/upsample_predictions.png")