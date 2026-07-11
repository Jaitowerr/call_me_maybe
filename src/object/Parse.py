import os
import sys
from typing import List, Any, Dict
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)
import json
from src.object.debug import Debug


class Config(BaseModel):
    input_path: str = Field(
        default="data/input/function_calling_tests.json"
    )
    input_func_path: str = Field(
        default="data/input/functions_definition.json"
    )
    output_path: str = Field(
        default="data/output/function_calling_results.json"
    )

    @field_validator("input_path", "output_path")
    @classmethod
    def check_json_extension(cls, v: str) -> str:
        if not v.endswith(".json"):
            raise ValueError(
                "The file must strictly end with '.json'. "
                f"Received: '{v}'"
            )
        return v

    @model_validator(mode="after")
    def check_paths_are_different(self) -> "Config":
        in_abs = os.path.abspath(self.input_path)
        out_abs = os.path.abspath(self.output_path)
        if in_abs == out_abs:
            raise ValueError(
                "Input and output paths cannot be the same."
            )
        return self

    @classmethod
    def parse_arguments(cls, args: List[str]) -> "Config":
        try:
            parsed_data = {}
            seen_flags = set()
            i = 0
            while i < len(args):
                if args[i] == "--input":
                    if "--input" in seen_flags:
                        raise ValueError(
                            "The --input flag is duplicated."
                        )
                    if i + 1 >= len(args) or args[i + 1].startswith("--"):
                        raise ValueError(
                            "The --input flag requires a file path."
                        )
                    parsed_data["input_path"] = args[i + 1]
                    seen_flags.add("--input")
                    i += 2

                elif args[i] == "--functions_definition":
                    if "--functions_definition" in seen_flags:
                        raise ValueError(
                            "The --functions_definition flag is duplicated."
                        )
                    if i + 1 >= len(args) or args[i + 1].startswith("--"):
                        raise ValueError(
                            "The --functions_definition flag requires"
                            " a file path."
                        )
                    parsed_data["input_func_path"] = args[i + 1]
                    seen_flags.add("--functions_definition")
                    i += 2
                elif args[i] == "--output":
                    if "--output" in seen_flags:
                        raise ValueError(
                            "The --output flag is duplicated."
                        )
                    if i + 1 >= len(args) or args[i + 1].startswith("--"):
                        raise ValueError(
                            "The --output flag requires a file path."
                        )
                    parsed_data["output_path"] = args[i + 1]
                    seen_flags.add("--output")
                    i += 2
                elif args[i] in ('--d', '--debug'):
                    if "debug" in seen_flags:
                        raise ValueError(
                            "The --debug flag is duplicated."
                        )
                    Debug().enable()
                    seen_flags.add("debug")
                    i += 1
                else:
                    raise ValueError(
                        f"Unrecognized argument: '{args[i]}'.\n"
                        "Usage: make run "
                        "[--input \"path/to/file.json\"] "
                        "[--functions_definition \"path/to/file.json\"] "
                        "[--output \"path/to/file.json\"]"
                    )

            return cls(**parsed_data)

        except (ValueError, ValidationError) as e:
            print("\033[1;31m\n[ARGUMENT ERROR]\033[0m")
            print("  -", e)
            sys.exit(1)

    def create_output_directory(self) -> None:
        """
        Verifies if the output directory exists and creates it if not.
        Handles permission errors and any other system exceptions.
        """
        try:
            output_dir = os.path.dirname(self.output_path)

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        except PermissionError:
            print("\033[1;31m\n[PERMISSION ERROR]\033[0m")
            print(
                "  - You do not have permission to create "
                f"the output directory: '{output_dir}'"
            )
            sys.exit(1)
        except Exception as e:
            print("\033[1;31m\n[DIRECTORY ERROR]\033[0m")
            print(
                "  - An unexpected error occurred while "
                f"creating '{output_dir}': {e}"
            )
            sys.exit(1)

    def write_output_json(self, result: List[Dict[str, Any]]) -> None:
        """
        Writes the result to the output JSON file.
        """
        try:
            if not isinstance(result, (dict, list)):
                raise TypeError(
                    "Result must be a dict or a list of dicts."
                )

            with open(self.output_path, 'w', encoding='utf-8') as file:
                json.dump(result, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print("[ERROR] Could not write the output file:", e)
            sys.exit(1)
