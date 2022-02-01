import sys

sys.path.append("glide-text2im")
from PIL import Image
import torch as th
from glide_text2im.download import load_checkpoint
import os
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)


def pred_to_pil(pred: th.Tensor) -> Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return Image.fromarray(reshaped.numpy())


def load_glide(
    model_name: str = "base",
    dropout: float = 0.1,
    timestep_respacing: str = "100",
    activation_checkpointing: bool = False,
):
    assert model_name in ["base", "upsample"], f"{model_name} is not a valid model name"
    if model_name == "base":
        options = model_and_diffusion_defaults()
    else:
        options = model_and_diffusion_defaults_upsampler()
        timestep_respacing = "fast27"
    
    options["use_fp16"] = True
    options["dropout"] = dropout
    options["timestep_respacing"] = timestep_respacing

    glide_model, glide_diffusion = create_model_and_diffusion(**options)
    glide_model.convert_to_fp16()
    glide_model.use_checkpoint = activation_checkpointing
    glide_model.load_state_dict(load_checkpoint(model_name, "cpu"))
    glide_model.eval()
    # Create CLIP model.
    return glide_model, glide_diffusion, options


def sample(
    model,
    eval_diffusion,
    options,
    side_x,
    side_y,
    prompt="",
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    clip_cond_fn=None,
):
    model.del_cache()
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, options["text_ctx"])
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options["text_ctx"]
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    full_batch_size = batch_size * 2
    samples = eval_diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, side_y, side_x),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=clip_cond_fn,
    )[:batch_size]
    model.del_cache()
    return samples


def sample_sr(
    model_up: th.nn.Module,
    diffusion_up: th.nn.Module,
    options_up: th.nn.Module,
    samples: th.Tensor,
    prompt: str,
    batch_size: int,
    upsample_temp: float = 0.997,
    device: th.device = th.device("cpu"),
    sr_x: int = 256,
    sr_y: int = 256,
):
    model_up.del_cache()
    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up["text_ctx"]
    )
    model_kwargs = dict(
        low_res=((samples + 1) * 127.5).round() / 127.5 - 1,
        tokens=th.tensor([tokens] * batch_size, device=device),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )
    up_shape = (batch_size, 3, sr_y, sr_x)
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
    )[:batch_size]
    model_up.del_cache()
    return up_samples