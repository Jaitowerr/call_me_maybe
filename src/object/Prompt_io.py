from __future__ import annotations
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel
import sys
import json
from pathlib import Path


class Prompt_io(BaseModel):
    """
    Representa un prompt del archivo de entrada.
    - prompt: texto de la pregunta
    - fn_name: nombre de la función seleccionada (rellenado más tarde)
    - args: diccionario de argumentos (rellenado más tarde)
    """
    prompt: str
    prompt_tk: Optional[List[int]] = None   # tokenización del texto del prompt.
    fn_name: Optional[str] = None
    args: Optional[Dict[str, Union[str, int, float, bool]]] = None

    @classmethod
    def load_prompts(cls, file_path: Path) -> List[Prompt_io]:
        """
        Lee el JSON de prompts y devuelve una lista de Prompt_io con solo el campo `prompt` relleno.
        Formato esperado: [ {"prompt": "texto 1"}, {"prompt": "texto 2"}, ... ]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("El archivo debe contener una lista de objetos.")

            results = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Elemento en posición {idx} no es un objeto.")

                if len(item) != 1:
                    raise ValueError(
                        f"Elemento en posición {idx} debe tener exactamente una clave.")

                key, value = next(iter(item.items()))
                if key != "prompt":
                    raise ValueError(
                        f"Clave en posición {idx} debe ser 'prompt', pero es '{key}'.")

                if not isinstance(value, str):
                    raise ValueError(
                        f"Valor en posición {idx} debe ser un string.")

                results.append(Prompt_io(prompt=value))

            return results

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {file_path}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: JSON inválido en {file_path}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error al cargar prompts: {e}", file=sys.stderr)
            sys.exit(1)


    def tokenize(self, encoder_func) -> List[int]:
        """Tokeniza el prompt usando la función dada (puede ser del SDK o mia)."""
        self.prompt_tk = encoder_func(self.prompt)

