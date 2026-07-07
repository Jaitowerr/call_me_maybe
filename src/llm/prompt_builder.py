from typing import List
from src.object.Prompt_io import Prompt_io
from src.object.Func_def import Func_def
import sys


class PromptBuilder():
    def __init__(self, encoder_func):
        self.encoder_func = encoder_func

    def build_tk(self, prompt: Prompt_io, func_tk: List[int]) -> List[int]:
        """
        Construye la lista de input_ids utilizando un delimitador estricto de bloque.
        Fuerza a la IA a generar el JSON completo (incluyendo llaves y claves) desde cero.
        """
        try:
            instructions = (
                "### System:\n"
                "You are a strict function-calling engine. Your output MUST be ONLY a single valid JSON object inside a <json> block. "
                "Do not write markdown fences (```json), do not write explanations, and do not include any prose.\n\n"
                
                "CRITICAL OUTPUT FORMAT:\n"
                "Your response must start exactly with the tag <json> followed by the complete JSON object, and end exactly with </json>.\n"
                "Inside the <json> block, you MUST generate the full JSON structure including the opening brace '{', the \"prompt\" key, and the closing brace '}'.\n\n"
                
                "THE JSON SCHEMA TO GENERATE:\n"
                "{\n"
                "  \"prompt\": \"<the exact verbatim user query string>\",\n"
                "  \"fn_name\": \"<function name>\",\n"
                "  \"args\": {\"parameter_name_1\": value_1, \"parameter_name_2\": value_2}\n"
                "}\n\n"
                
                "CRITICAL RULES FOR VALUES:\n"
                "- In the \"prompt\" field, you MUST copy the exact original text that the user wrote. Do not write the function name there.\n"
                "- In the \"args\" object, EVERY SINGLE NUMBER MUST BE A FLOAT WITH A DECIMAL POINT (e.g., 2.0, 3.0, 265.0, 345.0, 16.0, 144.0). INTEGERS ARE STRICTLY FORBIDDEN.\n\n"
                
                "Available functions:\n"
            )
            instructions_tk: List[int] = self.encoder_func(instructions)

            user_header = "\n### User:\n"
            user_header_tk: List[int] = self.encoder_func(user_header)

            user_prompt_tk: List[int] = prompt.get_prompt_tk()

            # --- RESPUESTA DIRECTA SIN PRE-LLENADO ---
            # Le damos paso al asistente de forma limpia con la etiqueta del bloque que queremos que abra.
            # Al ver '<json>', la inercia del modelo le obliga a escribir el '{' inmediatamente después 
            # y a rellenar el JSON entero con todas sus claves ("prompt", etc.) para poder cerrarlo con '</json>'.
            assistant_prefix = (
                "\n### Assistant:\n"
                "<json>"
            )
            assistant_prefix_tk: List[int] = self.encoder_func(assistant_prefix)

            # Concatenación de la secuencia de tokens
            full_tk = (
                instructions_tk
                + func_tk
                + user_header_tk
                + user_prompt_tk
                + assistant_prefix_tk
            )

            return full_tk

        except Exception as e:
            print(f"[ERROR] Fallo al construir tokens del prompt: {e}")
            sys.exit(1)


# CASI CASI CASI CASI
    # def build_tk(self, prompt: Prompt_io, func_tk: List[int]) -> List[int]:
    #     """
    #     Construye la lista de input_ids delimitando los roles de chat y 
    #     colocando al modelo exactamente en la apertura del JSON para que lo genere entero.
    #     """
    #     try:
    #         instructions = (
    #             "### System:\n"
    #             "You are a strict function-calling engine. Your response MUST be a single JSON object. "
    #             "Do not write markdown syntax (like ```json), headings, or explanations. "
    #             "You must start your response directly with the open curly brace '{' and include all fields.\n\n"
                
    #             "The JSON must follow this exact schema:\n"
    #             "{\n"
    #             "  \"prompt\": \"<the exact user query string>\",\n"
    #             "  \"fn_name\": \"<function name>\",\n"
    #             "  \"args\": {\"parameter_name_1\": value_1, \"parameter_name_2\": value_2}\n"
    #             "}\n\n"
                
    #             "CRITICAL RULES FOR ARGS:\n"
    #             "- \"args\" MUST be a valid key-value map object containing the exact parameter names from the function signature.\n"
    #             "- EVERY NUMBER IN ARGS MUST HAVE A DECIMAL POINT. DO NOT WRITE INTEGERS.\n"
    #             "  Example: If the value is 2, you MUST write 2.0. If the value is 5, you MUST write 5.0.\n\n"
                
    #             "Available functions:\n"
    #         )
    #         instructions_tk: List[int] = self.encoder_func(instructions)

    #         user_header = "\n### User:\n"
    #         user_header_tk: List[int] = self.encoder_func(user_header)

    #         user_prompt_tk: List[int] = prompt.get_prompt_tk()

    #         # --- ANCLAJE DE APERTURA ---
    #         # Forzamos la etiqueta de Assistant y abrimos la llave.
    #         # Al dejar el prompt clavado exactamente en el '{', obligamos a la IA a que su 
    #         # primer token autogenerado sea la clave '"prompt":' y rellene todo el JSON ella sola.
    #         assistant_prefix = (
    #             "\n### Assistant:\n"
    #             "{"
    #         )
    #         assistant_prefix_tk: List[int] = self.encoder_func(assistant_prefix)

    #         # Concatenación limpia de tokens de control
    #         full_tk = (
    #             instructions_tk
    #             + func_tk
    #             + user_header_tk
    #             + user_prompt_tk
    #             + assistant_prefix_tk
    #         )

    #         return full_tk

    #     except Exception as e:
    #         print(f"[ERROR] Fallo al construir tokens del prompt: {e}")
    #         sys.exit(1)




    # def build_tk(self, prompt: Prompt_io, func_tk: List[int]) -> List[int]:
    #     """
    #     Construye la lista de input_ids concatenando:
    #     1. Instrucciones del sistema
    #     2. Firmas de funciones (ya tokenizadas, recibidas como parámetro)
    #     3. Prompt del usuario
    #     4. Concatenar todo — el modelo genera el JSON completo desde '{'
    #     """
    #     try:
    #         instructions = (
    #             "You are a function-calling assistant.\n"
    #             "Your answer MUST be exactly one valid JSON object.\n"
    #             "Do not write explanations, Markdown, code fences, or any text before or after the JSON.\n\n"
    #             "The JSON must have exactly this structure:\n"
    #             "{\n"
    #             "    \"prompt\": \"<copy of the user query>\",\n"
    #             "    \"fn_name\": \"<function name>\",\n"
    #             "    \"args\": { ... }\n"
    #             "}\n\n"
    #             "Available functions:\n"
    #         )

    #         instructions_tk: List[int] = self.encoder_func(
    #             instructions)  # Tokenizamos las instrucciones

    #         separator = '\nUser query:\n'  # Separador entre funciones y el prompt del usuario
    #         separator_tk: List[int] = self.encoder_func(separator)

    #         # Prompt del usuario ya tokenizado
    #         user_prompt_tk: List[int] = prompt.get_prompt_tk()

    #         # Concatenación limpia: instrucciones + funciones + separador + query del usuario
    #         # El modelo debe generar el JSON completo desde '{' hasta '}'
    #         full_tk = (
    #             instructions_tk
    #             + func_tk
    #             + separator_tk
    #             + user_prompt_tk
    #         )

    #         return full_tk

    #     except Exception as e:
    #         print(f"[ERROR] Fallo al construir tokens del prompt: {e}")
    #         sys.exit(1)







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
