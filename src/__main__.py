
import sys
from src.object.Parse import Config
from src.object.Prompt_io import Prompt_io
from src.object.Func_def import Func_def
from pathlib import Path
from src.llm.wrapper import LLMWrapper


if __name__ == "__main__":
    rutas = Config.parse_arguments(sys.argv[1:])
    print("=== RUTAS ===")
    print(f"Input:  {rutas.input_path}")
    print(f"Output: {rutas.output_path}")
    print()

    input_path = Path(rutas.input_path)
    list_prompt = Prompt_io.load_prompts(input_path)

    print("=== PROMPTS CARGADOS ===")
    print(f"Total prompts: {len(list_prompt)}")
    for i, p in enumerate(list_prompt):
        print(f"  [{i}] prompt = {p.prompt!r}  fn_name={p.fn_name} args={p.args}")
    print()

    # Cargamos definiciones desde el archivo estándar (ajusta si quieres otro path)
    func_defs_path = Path("data/input/functions_definition.json")
    list_func_def = Func_def.load_func_def(func_defs_path)

    print("=== FUNCIONES CARGADAS ===")
    print(f"Total funciones: {len(list_func_def)}")
    for i, f in enumerate(list_func_def):
        print(f"  [{i}] name={f.name} desc={f.description!r}")
        print(f"       parameters={f.parameters}")
        print(f"       returns={f.returns}")
    print()
    
    # Creamso la ruta de saldia si lso objetos se han creado bien
    rutas.create_output_directory()
    print('Ruta -output creada\n')

    print('Creamos un str de cada funcion para tokeizar luego')
    # for _ in list_func_def: _.build_signature()   # La llevamos dentro de tokenize_signature
    ai_model = LLMWrapper()

    for _ in list_prompt: _.tokenize(ai_model.encode_text)
    for _ in list_func_def: _.tokenize_signature(ai_model.encode_text)


