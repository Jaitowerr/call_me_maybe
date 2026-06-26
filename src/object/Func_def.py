from typing import Dict
from pydantic import BaseModel

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
