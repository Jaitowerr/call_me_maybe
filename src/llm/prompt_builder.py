from typing import List, Callable
from src.object.Prompt_io import Prompt_io
import sys


class PromptBuilder():
    def __init__(self, encoder_func: Callable[[str], List[int]]) -> None:
        self.encoder_func = encoder_func

    def build_tk(self, prompt: Prompt_io, func_tk: List[int]) -> List[int]:
        """
        Builds input_ids using a strict block delimiter.
        Forces the AI to generate the full JSON from scratch.
        """
        try:
            instructions = (
                "### System:\n"
                "You are a strict function-calling engine. Your output MUST "
                "be ONLY a single valid JSON object inside a <json> block. "
                "Do not write markdown fences (```json), do not write "
                "explanations, and do not include any prose.\n\n"
                "CRITICAL OUTPUT FORMAT:\n"
                "Your response must start exactly with the tag <json> "
                "followed by the complete JSON object, and end exactly "
                "with </json>.\n"
                "Inside the <json> block, you MUST generate the full JSON "
                "structure including the opening brace '{', the \"prompt\" "
                "key, and the closing brace '}'.\n\n"
                "THE JSON SCHEMA TO GENERATE:\n"
                "{\n"
                "  \"prompt\": \"<the exact verbatim user query string>\",\n"
                "  \"fn_name\": \"<function name>\",\n"
                "  \"args\": {\"param_1\": value_1, \"param_2\": value_2}\n"
                "}\n\n"
                "CRITICAL RULES FOR VALUES:\n"
                "- In the \"prompt\" field, you MUST copy the exact original "
                "text that the user wrote. Do not write the function name.\n"
                "- In the \"args\" object, EVERY SINGLE NUMBER MUST BE A "
                "FLOAT WITH A DECIMAL POINT (e.g., 2.0, 16.0). "
                "INTEGERS ARE STRICTLY FORBIDDEN.\n\n"
                "Available functions:\n"
            )
            instructions_tk: List[int] = self.encoder_func(instructions)

            user_header = "\n### User:\n"
            user_header_tk: List[int] = self.encoder_func(user_header)

            user_prompt_tk: List[int] = prompt.get_prompt_tk()

            assistant_prefix = (
                "\n### Assistant:\n"
                "<json>"
            )
            assistant_prefix_tk: List[int] = self.encoder_func(
                assistant_prefix
            )

            full_tk = (
                instructions_tk
                + func_tk
                + user_header_tk
                + user_prompt_tk
                + assistant_prefix_tk
            )

            return full_tk

        except Exception as e:
            print(f"[ERROR] Failed to build prompt tokens: {e}")
            sys.exit(1)
