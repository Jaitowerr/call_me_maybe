from typing import List
from src.object.Prompt_io import Prompt_io
from src.object.Func_def import Func_def
import sys

class PromptBuilder():
    def __init__(self, encoder_func):
        self.encoder_func = encoder_func
    
    def build_tk(self, prompt: Prompt_io, funcions: List[Func_def]) -> List[int]:
        """
        Construye la lista de input_ids concatenando:
        1. Instrucciones del sistema
        2. Firmas de funciones
        3. Prompt del usuario
        4. Inicio forzado del JSON - suffix
        5. Concatenar todo
        """
        try:
            instructions = 'You are a function calling assistant. Respond ONLY with a JSON object.\nAvailable functions:\n'
            instructions_tk: List[int]= self.encoder_func(instructions)

            func_tk: List[int] = []
            for _ in funcions:
                func_tk.extend(_.get_signature_tk())

            separator = '\nUnser query: '
            separator_tk: List[int] = self.encoder_func(separator)

            prompt_tk:List[int] = prompt.get_prompt_tk()

            suffix = "\nResponse: {"
            suffix_tk = self.encoder_func(suffix)
            
            full_tk = instructions_tk + func_tk + separator_tk + prompt_tk + suffix_tk

            return full_tk
        except Exception as e:
            print(f"[ERROR] Fallo al construir tokens del prompt: {e}")
            sys.exit(1)
            