class StoryboardLLMInstruction:
    """Builds system/user prompts to feed into any LLM node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_concept": ("STRING", {"multiline": True, "default": "主人公が朝の街を歩き、不思議な扉を見つけて中へ入る"}),
                "character_description": ("STRING", {"multiline": True, "default": "the character in the reference image"}),
                "style": ("STRING", {"multiline": False, "default": "cinematic anime style, consistent character design"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("system_prompt", "user_prompt")
    FUNCTION = "build"
    CATEGORY = "Storyboard"

    def build(self, story_concept, character_description, style):
        system = (
            "You are a professional storyboard artist and prompt engineer for image generation models.\n"
            "Create exactly 4 sequential shots that tell the given story in order (beginning, development, turn, ending).\n"
            "Rules:\n"
            "- Write each shot prompt in English, 1-3 sentences.\n"
            "- Include: camera angle/shot size, character action and expression, location, lighting, and continuity from the previous shot.\n"
            "- Always refer to the protagonist as 'the character' (appearance comes from a reference image; do NOT redescribe face/outfit).\n"
            "- Keep location, time of day and props consistent across shots unless the story changes them.\n"
            "- Do NOT mention panel numbers, captions, speech bubbles or text in the image.\n"
            "Output ONLY valid JSON in this format, with no extra text:\n"
            '{"shots": ["shot 1 prompt", "shot 2 prompt", "shot 3 prompt", "shot 4 prompt"]}'
        )
        user = (
            f"Story: {story_concept}\n"
            f"Character: {character_description}\n"
            f"Visual style: {style}"
        )
        return (system, user)
