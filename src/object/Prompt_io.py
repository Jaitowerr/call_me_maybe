from __future__ import annotations
from typing import Optional, List, Callable
from pydantic import BaseModel
import sys
import json
from pathlib import Path


class Prompt_io(BaseModel):
    """
    Representa un prompt del archivo de entrada.
    - prompt: text de la pregunta
    - prompt_tk: text de la pregunta tokenizado

    """
    prompt: str
    prompt_tk: Optional[List[int]] = None   # token del text del prompt.

    @classmethod
    def load_prompts(cls, file_path: Path) -> List[Prompt_io]:
        """
        Lee el JSON de prompts y devuelve una lista de Prompt_io con solo
        el campo `prompt` relleno.
        Formato esperado: [ {"prompt": "text 1"}, {"prompt": "text 2"}, ... ]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(
                    "El archivo debe contener una lista de objetos.")

            results = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Elemento en posición {idx} no es un objeto.")

                if len(item) != 1:
                    raise ValueError(
                        f"Elemento en posición {idx} "
                        "debe tener exactamente una clave.")

                key, value = next(iter(item.items()))
                if key != "prompt":
                    raise ValueError(
                        f"Clave en posición {idx} debe ser 'prompt'"
                        ", pero es '{key}'.")

                if not isinstance(value, str):
                    raise ValueError(
                        f"Valor en posición {idx} debe ser un string.")
                if not value.strip():
                    print(
                        "\033[1;33m[INFO]\033[0m   "
                        f"Prompt vacío en posición {idx}, ignorado.")
                    continue

                results.append(Prompt_io(prompt=value))

            return results

        except FileNotFoundError:
            print(
                "Error: "
                f"No se encontró el archivo {file_path}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: JSON inválido en {file_path}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error al cargar prompts: {e}", file=sys.stderr)
            sys.exit(1)

    def tokenize(self, encoder_func: Callable[[str], List[int]]) -> None:
        """
        Tokeniza el prompt usando la función dada (puede ser del SDK o mia).
        Valida que el resultado sea una lista de enteros.
        """
        try:
            tokens = encoder_func(self.prompt)

            if not isinstance(tokens, list):
                raise Exception("El resultado no es una lista.")
            if not all(isinstance(_, int) for _ in tokens):
                raise Exception(
                    "La lista contiene elementos que no son enteros.")

            self.prompt_tk = tokens

        except Exception as e:
            print(f"Error crítico durante la tokenización del prompt: {e}")
            sys.exit(1)

    def get_prompt_tk(self) -> List[int]:
        if self.prompt_tk is None:
            raise ValueError("El prompt no está tokenizado.")
        return self.prompt_tk
