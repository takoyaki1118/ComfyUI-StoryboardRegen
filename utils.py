"""Shared helpers for the Storyboard nodes."""
import torch

import comfy.utils
import node_helpers
import nodes  # ComfyUI core nodes (for common_ksampler)


def encode_text(clip, text, guidance=None):
    tokens = clip.tokenize(text)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if guidance is not None and guidance > 0:
        cond = node_helpers.conditioning_set_values(cond, {"guidance": float(guidance)})
    return cond


def resize(img, w, h, method="lanczos"):
    """img: [B,H,W,C] -> [B,h,w,C]"""
    return comfy.utils.common_upscale(img.movedim(-1, 1), w, h, method, "disabled").movedim(1, -1)


def round_mult(v, mult=16):
    return max(mult, int(round(v / mult)) * mult)


def fit_megapixels(img, megapixels, mult=16):
    _, h, w, _ = img.shape
    scale = (megapixels * 1024 * 1024 / (w * h)) ** 0.5
    return resize(img, round_mult(w * scale, mult), round_mult(h * scale, mult))


def add_reference(cond, vae, ref_img):
    """Same mechanism as the built-in ReferenceLatent node."""
    lat = vae.encode(ref_img[:, :, :, :3])
    return node_helpers.conditioning_set_values(cond, {"reference_latents": [lat]}, append=True)


def empty_latent(model, w, h):
    ch = 16
    try:
        ch = model.get_model_object("latent_format").latent_channels
    except Exception:
        pass
    return {"samples": torch.zeros([1, ch, h // 8, w // 8])}


def split4(img, trim=0):
    """img [1,H,W,C] -> 4 tiles in order TL, TR, BL, BR."""
    _, H, W, _ = img.shape
    hh, hw = H // 2, W // 2
    tiles = []
    for (x, y) in [(0, 0), (hw, 0), (0, hh), (hw, hh)]:
        tiles.append(img[:, y + trim:y + hh - trim, x + trim:x + hw - trim, :])
    return tiles


def run_sampler(model, seed, steps, cfg, sampler, scheduler, pos, neg, latent, denoise):
    return nodes.common_ksampler(
        model, seed, steps, cfg, sampler, scheduler, pos, neg, latent, denoise=denoise
    )[0]
