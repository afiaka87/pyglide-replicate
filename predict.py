import torch as th
import cog
import util
import pathlib

import sys
sys.path.append("glide-text2im")

class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Initialize the model to cache it

    @cog.input(
        "prompt",
        type=str,
        default="",
        help="Text prompt to use.",
    )
    @cog.input(
        "batch_size",
        type=int,
        default=3,
        help="Batch size. Number of generations to predict",
        min=1,
        max=6,
    )
    @cog.input(
        "side_x",
        type=int,
        default=64,
        help="Must be multiple of 8. Going above 64 is not recommended. Actual image will be 4x larger. ",
        options=[32, 48, 64, 80, 96, 112, 128]
    )
    @cog.input(
        "side_y", 
        type=int,
        default=64,
        help="Must be multiple of 8. Going above 64 is not recommended. Actual image size will be 4x larger.",
        options=[32, 48, 64, 80, 96, 112, 128]
    )
    @cog.input(
        "upsample_stage",
        default=True,
        type=bool,
        help="Performs prompt-aware upsampling by 4x base resolution",
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
        default=0.998,
        help="Upsample temperature. Consider lowering to ~0.997 for blurry images with fewer artifacts.",
        options=[0.997, 0.998, 1.0]
    )
    @cog.input(
        "timestep_respacing",
        type=str,
        default="27",
        help="Number of timesteps to use for base model. Going above 150 has diminishing returns.",
        options=["5", "10", "15", "20", "25", "27", "fast27", "30", "35", "40", "45", "50", "75", "100", "125", "150"],
    )
    @cog.input(
        "sr_timestep_respacing",
        type=str,
        default="17",
        help="Number of timesteps to use for base model. Going above 150 has diminishing returns.",
        options=["5", "10", "15", "17", "19", "20", "21", "23", "25", "27", "fast27", "30", "35", "40", "45", "50", "75", "100", "125", "150"],
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
        guidance_scale,
        upsample_temp,
        timestep_respacing,
        sr_timestep_respacing,
        seed,
    ):
        prompt.replace("nazi", "").replace("swastika", "") # shrug
        th.manual_seed(seed)
        self.base_model, self.base_diffusion, self.base_options = util.load_glide(
            model_name="base", timestep_respacing=timestep_respacing
        )
        self.sr_model, self.sr_diffusion, self.sr_options = util.load_glide(
            model_name="upsample", timestep_respacing=sr_timestep_respacing
        )
        self.base_model.to("cuda")
        if upsample_stage:
            self.sr_model.to("cuda")
        with th.no_grad():
            print(f"Generating {side_x}x{side_y} samples with {timestep_respacing} timesteps using GLIDE-base-64px...")
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
                clip_cond_fn=None,
            )
            base_pil_images = util.pred_to_pil(base_samples)
            base_pil_images.save("/src/base_predictions.png")
            yield pathlib.Path("/src/base_predictions.png")

            if upsample_stage:
                print(f"Upsampling outputs from GLIDE-base {side_x}x{side_y} to {side_x*4}x{side_y*4} using {sr_timestep_respacing} timesteps...")
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