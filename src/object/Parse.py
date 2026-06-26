import os
import sys
from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

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
            