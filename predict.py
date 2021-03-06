import torch as th
import cog
import util
import pathlib

import sys
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

    @cog.input(
        "prompt",
        type=str,
        default="",
        help="Text prompt to use. Keep it simple/literal and avoid using poetic language (unlike CLIP).",
    )
    @cog.input(
        "batch_size",
        type=int,
        default=1,
        help="Batch size. Number of generations to predict",
        min=1,
        max=8,
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
        "upsample_stage",
        default=False,
        type=bool,
        help="If true, uses both the base and upsample models. If false, only the (finetuned) base model is used. This is useful for testing the upsampler, which is not finetuned.",
    )
    @cog.input(
        "upsample_temp",
        type=float,
        default=0.998,
        help="Upsample temperature. Consider lowering to ~0.997 for blurry images with fewer artifacts.",
        options=[0.996, 0.997, 0.998, 0.999, 1.0],
    )
    @cog.input(
        "timestep_respacing",
        type=str,
        default="35",
        help="Number of timesteps to use for base model PLMS sampling. Usually don't need more than 50.",
        options=["15", "17", "19", "21", "23", "25", "27", "30", "35", "40", "50", "100"]
    )
    @cog.input(
        "sr_timestep_respacing",
        type=str,
        default="17",
        help="Number of timesteps to use for upsample model PLMS sampling. Usually don't need more than 20.",
        options=["15", "17", "19", "21", "23", "25", "27"]
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
        upsample_stage,
        upsample_temp,
        guidance_scale,
        timestep_respacing,
        sr_timestep_respacing,
        seed,
    ):
        th.manual_seed(seed)
        # Run this again to change the model parameters
        self.base_model, self.base_diffusion, self.base_options = util.load_glide(
            model_name="base", timestep_respacing=timestep_respacing
        )
        self.sr_model, self.sr_diffusion, self.sr_options = util.load_glide(
            model_name="upsample", sr_timestep_respacing=sr_timestep_respacing
        )

        self.base_model.to("cuda")
        if upsample_stage:
            self.sr_model.to("cuda")
        with th.no_grad():
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
            )
            base_pil_images = util.pred_to_pil(base_samples)
            base_pil_images.save("/src/base_predictions.png")
            yield pathlib.Path("/src/base_predictions.png")

            if upsample_stage:
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