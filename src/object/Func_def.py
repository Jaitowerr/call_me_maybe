from typing import Dict, Optional, Callable, List
from pydantic import BaseModel
import sys
import json
from pathlib import Path


class Func_def(BaseModel):
    """
    Represents a function definition from function_definitions.json.
    parameters: {"a": {"type": "number"}, ...}
    returns: {"type": "number"}
    """
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]
    signature: Optional[str] = None
    signature_tk: Optional[List[int]] = None

    @classmethod
    def load_func_def(cls, file_path: Path) -> List["Func_def"]:
        """
        Reads functions_definitions.json and returns a list of Func_def.
        Expected format: list of objects with keys:
        name, description, parameters, returns
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(
                f"Error: definitions file not found: {file_path}",
                file=sys.stderr
            )
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(
                f"Error: invalid JSON in {file_path}: {e}",
                file=sys.stderr
            )
            sys.exit(1)

        if not isinstance(data, list):
            print(
                f"Error: definitions JSON must be a list in {file_path}",
                file=sys.stderr
            )
            sys.exit(1)

        definitions: List[Func_def] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                print(
                    f"Error: element #{idx} in {file_path} "
                    "is not a dict.",
                    file=sys.stderr
                )
                sys.exit(1)

            expected_keys = {"name", "description", "parameters", "returns"}
            missing = expected_keys - set(item.keys())
            extra = set(item.keys()) - expected_keys
            if missing:
                print(
                    f"Error: element #{idx} in {file_path} "
                    f"is missing keys: {missing}",
                    file=sys.stderr
                )
                sys.exit(1)
            if extra:
                print(
                    f"Error: element #{idx} in {file_path} "
                    f"has unexpected keys: {extra}",
                    file=sys.stderr
                )
                sys.exit(1)

            params = item["parameters"]
            if not isinstance(params, dict):
                print(
                    f"Error: 'parameters' of element #{idx} "
                    "must be a dict.",
                    file=sys.stderr
                )
                sys.exit(1)
            for pname, pspec in params.items():
                valid = (
                    isinstance(pspec, dict)
                    and "type" in pspec
                    and isinstance(pspec["type"], str)
                )
                if not valid:
                    print(
                        f"Error: parameter '{pname}' of element #{idx} "
                        "has invalid format "
                        "(expected {'type': '...'}).",
                        file=sys.stderr
                    )
                    sys.exit(1)

            returns = item["returns"]
            valid_returns = (
                isinstance(returns, dict)
                and "type" in returns
                and isinstance(returns["type"], str)
            )
            if not valid_returns:
                print(
                    f"Error: 'returns' of element #{idx} "
                    "has invalid format "
                    "(expected {'type': '...'}).",
                    file=sys.stderr
                )
                sys.exit(1)

            try:
                func = Func_def(**item)
            except Exception as e:
                print(
                    f"Error validating Func_def at element #{idx}: {e}",
                    file=sys.stderr
                )
                sys.exit(1)

            definitions.append(func)

        return definitions

    def build_signature(self) -> None:
        """Builds the textual signature of the function."""
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

    def tokenize_signature(
        self, encoder_func: Callable[[str], List[int]]
    ) -> None:
        """
        Tokenizes the signature using the given encoder function.
        Validates that the result is a list of integers.
        """
        try:
            if self.signature is None:
                self.build_signature()
                if self.signature is None:
                    raise ValueError("Signature could not be built (is None).")
            tokens = encoder_func(self.signature)
            if not isinstance(tokens, list):
                raise Exception("Result is not a list.")
            if not all(isinstance(_, int) for _ in tokens):
                raise Exception("List contains non-integer elements.")

            self.signature_tk = tokens
        except Exception as e:
            print(
                f"Critical error tokenizing signature of '{self.name}': {e}"
            )
            sys.exit(1)

    def get_signature_tk(self) -> List[int]:
        '''
        returns the tokenized signature attribute
        '''
        if self.signature_tk is None:
            raise ValueError(
                f"Signature of '{self.name}' is not tokenized."
            )
        return self.signature_tk

    def get_name_tk(
        self, encoder_func: Callable[[str], List[int]]
    ) -> List[int]:
        """
        Tokenizes and returns the function name as a list of IDs.
        """
        return encoder_func(self.name)
