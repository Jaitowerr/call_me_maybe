from typing import Dict, Any, Optional, Union
from pydantic import BaseModel

class Prompt_io(BaseModel):
    """
    Representa un prompt del archivo de entrada.
    - prompt: texto de la pregunta
    - fn_name: nombre de la función seleccionada (rellenado más tarde)
    - args: diccionario de argumentos (rellenado más tarde)
    """
    prompt: str
    fn_name: Optional[str] = None
    args: Optional[Dict[str, Union[str, int, float, bool]]] = None