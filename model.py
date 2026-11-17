import numpy as np
import torch
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline
from PIL import Image


def load_model(base_model, brushnet, device, model_cpu_offload=True, vae_tiling=False):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    brushnet = BrushNetModel.from_pretrained(str(brushnet), torch_dtype=dtype)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        str(base_model), brushnet=brushnet,
        torch_dtype=dtype, low_cpu_mem_usage=False,
    )
    if model_cpu_offload and device.type == "cuda":
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    if vae_tiling:
        pipe.enable_vae_tiling()
    return pipe


def generate_image(pipe, image, mask, seed, prompt, negative_prompt, steps, guidance, control):
    torch.manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=Image.fromarray(image),
        mask=Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB"),
        height=image.shape[0],
        width=image.shape[1],
        num_inference_steps=steps,
        guidance_scale=guidance,
        brushnet_conditioning_scale=control,
    ).images[0]
    return np.array(result.convert("RGB"))
