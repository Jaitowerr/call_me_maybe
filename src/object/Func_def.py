from typing import Dict, Optional
from pydantic import BaseModel
import sys
import json
from pathlib import Path
from typing import List

class Func_def(BaseModel):
    """
    Representa una definición de función según function_definitions.json.
    parameters: {"a": {"type": "number"}, ...}
    returns: {"type": "number"}
    """
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]] 
    returns: Dict[str, str]
    signature: str = None
    signature_tk : Optional[List[int]] = None	# tokenización del texto del prompt.

    @classmethod
    def load_func_def(cls, file_path: Path) -> List["Func_def"]:
        """
        Lee functions_definitions.json y devuelve una lista de Func_def.
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
    
    def build_signature(self):
        """Construye la firma textual de la función."""
        parameters = []
        for key, value in self.parameters.items():
            value_type = value['type']
            text = f'{key} ({value_type})'
            parameters.append(text)
        
        parameters_join = ', '.join(parameters)

        self.signature = (
            f'- {self.name}: {self.description}. '
            f'Parameters: {parameters_join}. '
            f"Returns: {self.returns['type']}"
        )    
        # print(self.signature)   # eliminar despues

    def tokenize_signature(self, encoder_func):
        """
        Tokeniza la firma usando la función dada.
        Valida que el resultado sea una lista de enteros.
        """
        if self.signature is None:
            self.build_signature()
        try:
            tokens = encoder_func(self.signature)
            if not isinstance(tokens, list):
                raise Exception("El resultado no es una lista.")
            if not all(isinstance(_, int) for _ in tokens):
                raise Exception("La lista contiene elementos que no son enteros.")

            self.signature_tk = tokens
        except Exception as e:
            print(f"Error crítico durante la tokenización de la firma de '{self.name}': {e}")
            sys.exit(1)
            

    def get_signature_tk(self) -> List[int]:
        if self.signature_tk is None:
            raise ValueError(f"La firma de {self.name} no está tokenizada.")
        return self.signature_tk
    
    def get_name_tk(self, encoder_func) -> List[int]:
        """
        Tokeniza y devuelve el nombre de la función como lista de IDs.
        """
        return encoder_func(self.name)
