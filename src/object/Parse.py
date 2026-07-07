import os
import sys
from typing import List, Any, Dict
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
import json

class Config(BaseModel):
    input_path: str = Field(default="data/input/function_calling_tests.json")
    output_path: str = Field(default="data/output/function_calling_results.json")

    @field_validator("input_path", "output_path")
    @classmethod
    def check_json_extension(cls, v: str) -> str:
        if not v.endswith(".json"):
            raise ValueError(f"El archivo debe terminar estrictamente en '.json'. Recibido: '{v}'")
        return v

    @model_validator(mode="after")
    def check_paths_are_different(self) -> "Config":
        if os.path.abspath(self.input_path) == os.path.abspath(self.output_path):
            raise ValueError("La ruta de entrada (--input) y de salida (--output) no pueden ser iguales.")
        return self

    @classmethod
    def parse_arguments(cls, args: List[str]) -> "Config":
        try:
            parsed_data = {}
            i = 0
            while i < len(args):
                if args[i] == "--input":
                    if i + 1 >= len(args) or args[i+1].startswith("--"):
                        raise ValueError("La bandera --input requiere que especifiques una ruta de archivo.")
                    parsed_data["input_path"] = args[i+1]
                    i += 2
                elif args[i] == "--output":
                    if i + 1 >= len(args) or args[i+1].startswith("--"):
                        raise ValueError("La bandera --output requiere que especifiques una ruta de archivo.")
                    parsed_data["output_path"] = args[i+1]
                    i += 2
                else:
                    raise ValueError(
                        f"Argumento no reconocido: '{args[i]}'.\n"
                        "Uso permitido: make run [--input \"ruta/archivo.json\"] [--output \"ruta/archivo.json\"]"
                    )

            # Llama a Pydantic para fabricar el objeto y ejecutar los validadores automáticos
            return cls(**parsed_data)

        except (ValueError, ValidationError) as e:
            print(f"\033[1;31m\n[ERROR DE ARGUMENTOS]\033[0m")
            print(f'  - ', e)
            sys.exit(1)
            
    def create_output_directory(self) -> None:
        """
        Verifica si existe el directorio de la ruta de salida y lo crea si no existe.
        Gestiona errores de permisos o cualquier otra excepción del sistema.
        """
        try:
            # Extraemos solo la ruta sin el .json)
            output_dir = os.path.dirname(self.output_path)
            # print(output_dir)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
        except PermissionError:
            print(f"\033[1;31m\n[ERROR DE PERMISOS]\033[0m")
            print(f"  - No tienes permisos para crear el directorio de salida: '{output_dir}'")
            sys.exit(1)
        except Exception as e:
            print(f"\033[1;31m\n[ERROR AL CREAR DIRECTORIO]\033[0m")
            print(f"  - Ocurrió un error inesperado al crear '{output_dir}': {e}")
            sys.exit(1)
    
    def write_output_json(self, result: List[Dict[str, Any]]) -> None:
        """
        Escribe el resultado en el fichero JSON de salida.
        """
        try:
            if not isinstance(result,(dict, list)):
                raise TypeError("El resultado debe ser un dict o una lista de dicts.")

            with open(self.output_path, 'w', encoding = 'utf-8') as file:
                json.dump(result, file, indent=4, ensure_ascii=False)   #ensure_ascii conserva caracteres UTF-8 correctamente.
        except Exception as e:
            print(f"[ERROR] No se pudo escribir el fichero de salida: {e}")
            sys.exit(1)
    