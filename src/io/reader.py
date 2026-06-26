import sys
import json
from pathlib import Path
from typing import List
from ..object.Prompt_io import Prompt_io
from ..object.Func_def import Func_def


def load_prompts(file_path: Path) -> List[Prompt_io]:
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
                raise ValueError(f"Elemento en posición {idx} no es un objeto.")
            
            if len(item) != 1:
                raise ValueError(f"Elemento en posición {idx} debe tener exactamente una clave.")
            
            key, value = next(iter(item.items()))
            if key != "prompt":
                raise ValueError(f"Clave en posición {idx} debe ser 'prompt', pero es '{key}'.")
            
            if not isinstance(value, str):
                raise ValueError(f"Valor en posición {idx} debe ser un string.")

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


def load_function_definitions(file_path: Path) -> List[Func_def]:
    """
    Lee function_definitions.json y devuelve una lista de Func_def.
    Formato esperado: lista de objetos con claves: name, description, parameters, returns
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: no se encontró el archivo de definiciones: {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: JSON inválido en {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(f"Error: el JSON de definiciones debe ser una lista en {file_path}", file=sys.stderr)
        sys.exit(1)

    definitions: List[Func_def] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Error: elemento #{idx} en {file_path} no es un objeto/dict.", file=sys.stderr)
            sys.exit(1)

        expected_keys = {"name", "description", "parameters", "returns"}
        missing = expected_keys - set(item.keys())
        extra = set(item.keys()) - expected_keys
        if missing:
            print(f"Error: elemento #{idx} en {file_path} falta claves: {missing}", file=sys.stderr)
            sys.exit(1)
        if extra:
            print(f"Error: elemento #{idx} en {file_path} tiene claves inesperadas: {extra}", file=sys.stderr)
            sys.exit(1)

        # Validamos parámetros: debe ser dict de dicts con clave 'type'
        params = item["parameters"]
        if not isinstance(params, dict):
            print(f"Error: 'parameters' del elemento #{idx} debe ser un dict.", file=sys.stderr)
            sys.exit(1)
        for pname, pspec in params.items():
            if not isinstance(pspec, dict) or "type" not in pspec or not isinstance(pspec["type"], str):
                print(f"Error: parámetro '{pname}' del elemento #{idx} tiene formato inválido (debe ser {{'type': '...' }}).", file=sys.stderr)
                sys.exit(1)

        # Validamos returns
        returns = item["returns"]
        if not isinstance(returns, dict) or "type" not in returns or not isinstance(returns["type"], str):
            print(f"Error: 'returns' del elemento #{idx} tiene formato inválido (debe ser {{'type': '...' }}).", file=sys.stderr)
            sys.exit(1)

        # Si todo OK, validamos con Pydantic (lanzará error si algo no cuadra)
        try:
            func = Func_def(**item)
        except Exception as e:
            print(f"Error al validar Func_def en elemento #{idx}: {e}", file=sys.stderr)
            sys.exit(1)

        definitions.append(func)

    return definitions

# def load_function_definitions(file_path: Path) -> List[Func_def]:
#     """
#     Lee function_definitions.json y devuelve una lista de Func_def.
#     Formato esperado: lista de objetos con claves: name, description, parameters, returns
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         if not isinstance(data, list):
#             raise ValueError("El archivo debe contener una lista de objetos.")

#         definitions = []
#         for idx, item in enumerate(data):
#             if not isinstance(item, dict):
#                 raise ValueError(f"Elemento en posición {idx} no es un objeto.")
#             definitions.append(Func_def(**item))

#         return definitions

#     except FileNotFoundError:
#         print(f"Error: No se encontró el archivo {file_path}", file=sys.stderr)
#         sys.exit(1)
#     except json.JSONDecodeError as e:
#         print(f"Error: JSON inválido en {file_path}: {e}", file=sys.stderr)
#         sys.exit(1)
#     except Exception as e:
#         print(f"Error al cargar definiciones: {e}", file=sys.stderr)
#         sys.exit(1)
