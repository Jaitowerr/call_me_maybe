import sys
from src.object.Parse import Config
from src.object.Prompt_io import Prompt_io
from src.object.Func_def import Func_def
from pathlib import Path
from src.llm.wrapper import LLMWrapper
from src.llm.prompt_builder import PromptBuilder
from typing import List
from src.object.debug import Debug


if __name__ == "__main__":
    rutas = Config.parse_arguments(sys.argv[1:])
    dprint = Debug().dprint

    print("\n\033[1;36m[INFO]\033[0m Rutas configuradas.")
    print(f"       Input:  {rutas.input_path}")
    print(f"       Output: {rutas.output_path}")

    input_path = Path(rutas.input_path)
    list_prompt = Prompt_io.load_prompts(input_path)

    print(
        f"\n\n\033[1;32m[OK]\033[0m   Prompts cargados: {len(list_prompt)} entradas.")
    dprint(f"    [DEBUG] Total prompts: {len(list_prompt)}")
    for i, p in enumerate(list_prompt):
        dprint(f"  [{i}] {p.prompt!r}")

    func_defs_path = Path("data/input/functions_definition.json")
    list_func_def = Func_def.load_func_def(func_defs_path)

    print(
        f"\n\n\033[1;32m[OK]\033[0m   Funciones cargadas: {len(list_func_def)} definiciones.")
    dprint(f"    [DEBUG] Total funciones: {len(list_func_def)}")
    for i, f in enumerate(list_func_def):
        dprint(f"  [{i}] name={f.name} desc={f.description!r}")
        dprint(f"       parameters={f.parameters}")
        dprint(f"       returns={f.returns}")

    rutas.create_output_directory()
    print(f"\n\n\033[1;32m[OK]\033[0m   Directorio de salida preparado.\n")

    ai_model = LLMWrapper()
    print(f"\n\n\033[1;32m[OK]\033[0m   Modelo LLM inicializado.")

    for _ in list_prompt:
        _.tokenize(ai_model.encode_text)
    dprint('    \n---> tokeniazación str prompt Prompt_io')
    for _ in list_func_def:
        _.tokenize_signature(ai_model.encode_text)
    dprint('    ---> tokeniazación str de Func_def, funciones tokenizadas')

    builder_tk = PromptBuilder(ai_model.encode_text)
    print(f"\n\n\033[1;32m[OK]\033[0m   Prompt Builder tokenizado.")

    for prompt in list_prompt:
        full_prompt_tk = builder_tk.build_tk(prompt, list_func_def)
        dprint(f"    ✅ Tokens generados: {len(full_prompt_tk)}")
        dprint(
            f"        - PromptBuilder construyó correctamente el prompt : {prompt.prompt}")
    dprint()

    print(f"\n\n\033[1;36m[INFO]\033[0m Cargando vocabulario y logits...")

    ai_model.load_vocab()
    print(f"\n\n\033[1;32m[OK]\033[0m   Vocabulario cargado.")
    dprint(f"\n    [DEBUG] Total tokens: {len(ai_model.id_to_tk_str)}")

    list_func_def_tk: List[int] = []
    for f in list_func_def:
        list_func_def_tk.extend(f.get_signature_tk())

    all_full_tk: List[List[int]] = []
    for prompt in list_prompt:
        prompt_full_tokenizado = builder_tk.build_tk(prompt, list_func_def_tk)
        all_full_tk.append(prompt_full_tokenizado)

    result = []
    print(
        f"\n\n\033[1;36m[INFO]\033[0m Generando respuestas restringidas...\n")
    for full_tk, prompt in zip(all_full_tk, list_prompt):
        answer = ai_model.respuesta_ia(
            full_tk, prompt.get_prompt_tk(), list_func_def)
        result.append(answer)
        print(f"\033[1;32m[OK]\033[0m   Prompt procesado.")
        dprint(f"    [DEBUG] Respuesta: {answer}")
    rutas.write_output_json(result)

    print(
        f"\n\n\n\033[1;32m[OK]\033[0m   Resultados guardados en: {rutas.output_path}")
    print("\n\n\033[1;35m[SUCCESS]\033[0m Ejecución terminada con éxito.\n")
