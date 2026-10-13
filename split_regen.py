import torch

import comfy.samplers
import comfy.utils

from .utils import (encode_text, add_reference, fit_megapixels, resize,
                    round_mult, split4, run_sampler)


class StoryboardSplitRegen:
    """Split the 2x2 grid into 4 panels and regenerate each one in high resolution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE",),
                "storyboard_grid": ("IMAGE",),
                "shot_1": ("STRING", {"forceInput": True}),
                "shot_2": ("STRING", {"forceInput": True}),
                "shot_3": ("STRING", {"forceInput": True}),
                "shot_4": ("STRING", {"forceInput": True}),
                "long_side": ("INT", {"default": 1536, "min": 512, "max": 4096, "step": 16}),
                "border_trim": ("INT", {"default": 6, "min": 0, "max": 128}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.01}),
                "ref_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "guidance": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            },
            "optional": {
                "style": ("STRING", {"multiline": False, "default": "cinematic anime style, highly detailed, sharp"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("hq_batch", "hq_1", "hq_2", "hq_3", "hq_4", "raw_tiles_batch")
    FUNCTION = "run"
    CATEGORY = "Storyboard"

    def run(self, model, clip, vae, reference_image, storyboard_grid,
            shot_1, shot_2, shot_3, shot_4, long_side, border_trim, denoise,
            ref_megapixels, seed, steps, cfg, guidance, sampler_name, scheduler,
            style="", negative_prompt=""):

        ref = fit_megapixels(reference_image[:1], ref_megapixels)
        tiles = split4(storyboard_grid[:1], border_trim)
        shots = [shot_1, shot_2, shot_3, shot_4]

        _, th, tw, _ = tiles[0].shape
        if tw >= th:
            W, H = long_side, long_side * th / tw
        else:
            H, W = long_side, long_side * tw / th
        W, H = round_mult(W), round_mult(H)

        neg = add_reference(encode_text(clip, negative_prompt or ""), vae, ref)

        results, raws = [], []
        pbar = comfy.utils.ProgressBar(4)
        for i, (tile, shot) in enumerate(zip(tiles, shots)):
            up = resize(tile, W, H, "lanczos")  # pixel upscale first, then low-denoise regen
            raws.append(up)

            prompt = (
                f"{shot} The same character as in the reference image, "
                f"identical face, hairstyle and outfit. {style}"
            )
            pos = add_reference(encode_text(clip, prompt, guidance), vae, ref)

            lat = {"samples": vae.encode(up[:, :, :, :3])}
            out = run_sampler(model, seed + i, steps, cfg, sampler_name, scheduler,
                              pos, neg, lat, denoise)
            results.append(vae.decode(out["samples"])[:, :, :, :3])
            pbar.update(1)

        return (torch.cat(results, dim=0), results[0], results[1], results[2], results[3],
                torch.cat(raws, dim=0))
