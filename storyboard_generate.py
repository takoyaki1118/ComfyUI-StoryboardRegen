import comfy.samplers

from .utils import encode_text, add_reference, fit_megapixels, empty_latent, run_sampler


class StoryboardGenerate:
    """One-pass 2x2 storyboard generation using a character reference image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE",),
                "grid_prompt": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 1344, "min": 512, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1344, "min": 512, "max": 4096, "step": 16}),
                "ref_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "guidance": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("storyboard_grid",)
    FUNCTION = "generate"
    CATEGORY = "Storyboard"

    def generate(self, model, clip, vae, reference_image, grid_prompt, width, height,
                 ref_megapixels, seed, steps, cfg, guidance, sampler_name, scheduler,
                 negative_prompt=""):
        ref = fit_megapixels(reference_image[:1], ref_megapixels)
        pos = add_reference(encode_text(clip, grid_prompt, guidance), vae, ref)
        neg = add_reference(encode_text(clip, negative_prompt or ""), vae, ref)

        latent = empty_latent(model, width, height)
        out = run_sampler(model, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, 1.0)
        img = vae.decode(out["samples"])
        return (img[:, :, :, :3],)
