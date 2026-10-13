from .llm_instruction import StoryboardLLMInstruction
from .prompt_parser import StoryboardPromptParser
from .storyboard_generate import StoryboardGenerate
from .split_regen import StoryboardSplitRegen

NODE_CLASS_MAPPINGS = {
    "StoryboardLLMInstruction": StoryboardLLMInstruction,
    "StoryboardPromptParser": StoryboardPromptParser,
    "StoryboardGenerate": StoryboardGenerate,
    "StoryboardSplitRegen": StoryboardSplitRegen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryboardLLMInstruction": "Storyboard LLM Instruction",
    "StoryboardPromptParser": "Storyboard Prompt Parser",
    "StoryboardGenerate": "Storyboard Generate (2x2, Char Ref)",
    "StoryboardSplitRegen": "Storyboard Split & Regen HQ",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
