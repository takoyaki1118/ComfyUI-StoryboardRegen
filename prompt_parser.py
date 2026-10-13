import json
import re


def parse_shots(text):
    shots = []
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            arr = data.get("shots") if isinstance(data, dict) else None
            if isinstance(arr, list):
                shots = [str(s).strip() for s in arr if str(s).strip()]
        except Exception:
            pass
    if len(shots) < 4:  # fallback: numbered / "Shot N:" lines
        shots = []
        for line in text.splitlines():
            mm = re.match(r"^(?:shot\s*)?(\d+)[\.\):：\-\s]+(.+)$", line.strip(), re.I)
            if mm:
                shots.append(mm.group(2).strip())
    if len(shots) < 4:  # last resort: non-empty lines
        shots = [l.strip() for l in text.splitlines() if l.strip()]
    shots = shots[:4]
    while len(shots) < 4:
        shots.append(shots[-1] if shots else "the character standing")
    return shots


class StoryboardPromptParser:
    """LLM text -> 4 shot prompts + 2x2 grid prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_text": ("STRING", {"forceInput": True}),
                "style": ("STRING", {"multiline": False, "default": "cinematic anime style, consistent character design"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("grid_prompt", "shot_1", "shot_2", "shot_3", "shot_4")
    FUNCTION = "parse"
    CATEGORY = "Storyboard"

    def parse(self, llm_text, style):
        s = parse_shots(llm_text)
        grid = (
            "A single image containing a 2x2 grid storyboard with four equally sized panels "
            "separated by thin white gutters. The same character from the reference image appears in every panel "
            "with identical face, hairstyle, outfit and proportions. The four panels are sequential shots of one story. "
            f"Top-left panel: {s[0]} "
            f"Top-right panel: {s[1]} "
            f"Bottom-left panel: {s[2]} "
            f"Bottom-right panel: {s[3]} "
            f"Style: {style}. No text, no captions."
        )
        return (grid, s[0], s[1], s[2], s[3])
