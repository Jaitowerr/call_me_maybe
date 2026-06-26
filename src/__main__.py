import sys
from src.object.Parse import Config
from src.io.reader import load_prompts, load_function_definitions
from pathlib import Path


if __name__ == "__main__":
    rutas = Config.parse_arguments(sys.argv[1:])
    print("=== RUTAS ===")
    print(f"Input:  {rutas.input_path}")
    print(f"Output: {rutas.output_path}")
    print()

    input_path = Path(rutas.input_path)
    list_prompt = load_prompts(input_path)

    print("=== PROMPTS CARGADOS ===")
    print(f"Total prompts: {len(list_prompt)}")
    for i, p in enumerate(list_prompt):
        print(f"  [{i}] prompt = {p.prompt!r}  fn_name={p.fn_name} args={p.args}")
    print()

    # Cargamos definiciones desde el archivo estándar (ajusta si quieres otro path)
    func_defs_path = Path("data/input/functions_definition.json")
    list_func_def = load_function_definitions(func_defs_path)

    print("=== FUNCIONES CARGADAS ===")
    print(f"Total funciones: {len(list_func_def)}")
    for i, f in enumerate(list_func_def):
        print(f"  [{i}] name={f.name} desc={f.description!r}")
        print(f"       parameters={f.parameters}")
        print(f"       returns={f.returns}")
    print()