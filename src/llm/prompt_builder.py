from typing import List
from src.object.Prompt_io import Prompt_io
from src.object.Func_def import Func_def
import sys


class PromptBuilder():
    def __init__(self, encoder_func):
        self.encoder_func = encoder_func

    def build_tk(self, prompt: Prompt_io, func_tk: List[int]) -> List[int]:
        """
        Construye la lista de input_ids concatenando:
        1. Instrucciones del sistema
        2. Firmas de funciones (ya tokenizadas, recibidas como parámetro)
        3. Prompt del usuario
        4. Concatenar todo — el modelo genera el JSON completo desde '{'
        """
        try:
            instructions = (
                "You are a function-calling assistant.\n"
                "Your answer MUST be exactly one valid JSON object.\n"
                "Do not write explanations, Markdown, code fences, or any text before or after the JSON.\n\n"
                "The JSON must have exactly this structure:\n"
                "{\n"
                "    \"prompt\": \"<copy of the user query>\",\n"
                "    \"fn_name\": \"<function name>\",\n"
                "    \"args\": { ... }\n"
                "}\n\n"
                "Available functions:\n"
            )

            instructions_tk: List[int] = self.encoder_func(
                instructions)  # Tokenizamos las instrucciones

            separator = '\nUser query:\n'  # Separador entre funciones y el prompt del usuario
            separator_tk: List[int] = self.encoder_func(separator)

            # Prompt del usuario ya tokenizado
            user_prompt_tk: List[int] = prompt.get_prompt_tk()

            # Concatenación limpia: instrucciones + funciones + separador + query del usuario
            # El modelo debe generar el JSON completo desde '{' hasta '}'
            full_tk = (
                instructions_tk
                + func_tk
                + separator_tk
                + user_prompt_tk
            )

            return full_tk

        except Exception as e:
            print(f"[ERROR] Fallo al construir tokens del prompt: {e}")
            sys.exit(1)

    # def build_tk(self, prompt: Prompt_io, funcions: List[Func_def]) -> List[int]:
    #     """
    #     Construye la lista de input_ids concatenando:
    #     1. Instrucciones del sistema
    #     2. Firmas de funciones
    #     3. Prompt del usuario
    #     4. Inicio forzado del JSON - suffix
    #     5. Concatenar todo
    #     """
    #     try:
    #         # instructions = 'You are a function calling assistant. Respond ONLY with a JSON object.\nAvailable functions:\n'
    #         # instructions = '''You are a function-calling assistant. Respond ONLY with a single JSON object that follows this EXACT schema:
    #         # {
    #         # "prompt": "<the original user prompt as a string>",
    #         # "fn_name": "<one of the available function names exactly>",
    #         # "args": { ... }
    #         # }

    #         # Rules:
    #         # 1. Always include the keys "prompt", "fn_name", and "args" in that order.
    #         # 2. "prompt": copy verbatim the user prompt (string).
    #         # 3. "fn_name": must be exactly one of the available function names (for example: fn_add_numbers, fn_greet, fn_reverse_string, fn_get_square_root, fn_substitute_string_with_regex).
    #         # 4. "args": must be a JSON object whose keys and value types match the selected function's parameter specification. Numbers must be numeric (no quotes); strings must use double quotes.
    #         # 5. Don't include additional keys.
    #         # 6. No surrounding text, explanation, or punctuation — output must be the JSON object only.

    #         # Example:
    #         # {
    #         # "prompt": "What is the sum of 2 and 3?",
    #         # "fn_name": "fn_add_numbers",
    #         # "args": {"a": 2.0, "b": 3.0}
    #         # }
    #         # '''

    #         # instructions = (
    #         #     "You are a function-calling assistant. "
    #         #     "Respond ONLY with a single JSON object.\n\n"
    #         #     "Available functions:\n"
    #         # )

    #         instructions = (
    #             "You are a function-calling assistant. Respond ONLY with a single JSON object.\n"
    #             "The JSON MUST contain these keys in order: \"prompt\", \"fn_name\", \"args\".\n\n"
    #             "Example:\n"
    #             "{\n"
    #             "  \"prompt\": \"What is the sum of 2 and 3?\",\n"
    #             "  \"fn_name\": \"fn_add_numbers\",\n"
    #             "  \"args\": {\"a\": 2.0, \"b\": 3.0}\n"
    #             "}\n\n"
    #             "Available functions:\n"
    #         )

    #         instructions_tk: List[int] = self.encoder_func(instructions)  # Tokenizamos

    #         func_tk: List[int] = []
    #         for _ in funcions:  # Traemos todos las funciones ya tokenizadas
    #             func_tk.extend(_.get_signature_tk())

    #         separator = '\nUser query: '  # Separador
    #         separator_tk: List[int] = self.encoder_func(separator)

    #         # Prompt del usuario
    #         user_prompt_tk: List[int] = prompt.get_prompt_tk()

    #         # suffix = "\nResponse: {"		#Le decimos como debe empezar por si acaso:
    #         # Forzamos el inicio del JSON con solo la llave de apertura.
    #         # El modelo debe generar TODO: "prompt", "fn_name", "args" y sus valores.
    #         # suffix = '\nResponse: {'
    #         # suffix_tk = self.encoder_func(suffix)

    #         # ✅ Prompt literal otra vez (para que el modelo lo "copie" en el JSON)
    #         # (mantenido como variable pero no usado en Opción A — el modelo lo genera solo)
    #         prompt_literal_tk: List[int] = prompt.get_prompt_tk()

    #         # ✅ Cierre del campo prompt y apertura de fn_name
    #         # (mantenido como variable pero no usado en Opción A — el modelo lo genera solo)
    #         # fn_name_start = '", "fn_name": "'
    #         # fn_name_start_tk: List[int] = self.encoder_func(fn_name_start)

    #         # ✅ Concatenación limpia: el modelo recibe contexto hasta '{' y genera el resto
    #         full_tk = (
    #             instructions_tk
    #             + func_tk
    #             + separator_tk
    #             + user_prompt_tk
    #             + suffix_tk
    #         )

    #         return full_tk

    #     except Exception as e:
    #         print(f"[ERROR] Fallo al construir tokens del prompt: {e}")
    #         sys.exit(1)
