import sys
from pathlib import Path
from typing import List
from src.object.Parse import Config
from src.object.Prompt_io import Prompt_io
from src.object.Func_def import Func_def
from src.llm.wrapper import LLMWrapper
from src.llm.prompt_builder import PromptBuilder
from src.object.debug import Debug
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


if __name__ == "__main__":
    rutas = Config.parse_arguments(sys.argv[1:])
    dprint = Debug().dprint

    print("\n\033[1;36m[INFO]\033[0m Paths configured.")
    print(f"       Input:  {rutas.input_path}")
    print(f"       Output: {rutas.output_path}")

    input_path = Path(rutas.input_path)
    list_prompt = Prompt_io.load_prompts(input_path)

    print(
        "\n\n\033[1;32m[OK]\033[0m   Prompts loaded:"
        f" {len(list_prompt)} entries.")
    dprint(f"    [DEBUG] Total prompts: {len(list_prompt)}")
    for i, p in enumerate(list_prompt):
        dprint(f"  [{i}] {p.prompt!r}")

    func_defs_path = Path("data/input/functions_definition.json")
    list_func_def = Func_def.load_func_def(func_defs_path)

    print(
        "\n\n\033[1;32m[OK]\033[0m   Functions loaded:"
        f" {len(list_func_def)} definitions.")
    dprint(f"    [DEBUG] Total functions: {len(list_func_def)}")
    for i, f in enumerate(list_func_def):
        dprint(f"  [{i}] name={f.name} desc={f.description!r}")
        dprint(f"       parameters={f.parameters}")
        dprint(f"       returns={f.returns}")

    rutas.create_output_directory()
    print("\n\n\033[1;32m[OK]\033[0m   Output directory ready.\n")

    ai_model = LLMWrapper("Qwen/Qwen3-0.6B", Small_LLM_Model)
    print("\n\n\033[1;32m[OK]\033[0m   LLM Model initialized.")

    for prm_io in list_prompt:
        prm_io.tokenize(ai_model.encode_text)
    dprint('    \n---> tokenization of prompt strings (Prompt_io)')
    for fn_df in list_func_def:
        fn_df.tokenize_signature(ai_model.encode_text)
    dprint('    ---> tokenization of function strings (Func_def)')

    builder_tk = PromptBuilder(ai_model.encode_text)
    print("\n\n\033[1;32m[OK]\033[0m   Prompt Builder tokenized.")

    list_func_def_tk: List[int] = []
    for f in list_func_def:
        list_func_def_tk.extend(f.get_signature_tk())

    for prompt in list_prompt:
        full_prompt_tk = builder_tk.build_tk(prompt, list_func_def_tk)
        dprint(f"    ✅ Tokens generated: {len(full_prompt_tk)}")
        dprint(
            "        - PromptBuilder correctly built the prompt: "
            f"{prompt.prompt}"
        )
    dprint()

    print("\n\n\033[1;36m[INFO]\033[0m Loading vocabulary and logits...")

    ai_model.load_vocab()
    print("\n\n\033[1;32m[OK]\033[0m   Vocabulary loaded.")
    dprint(f"\n    [DEBUG] Total tokens: {len(ai_model.id_to_tk_str)}")

    all_full_tk: List[List[int]] = []
    for prompt in list_prompt:
        prompt_full_tokenizado = builder_tk.build_tk(prompt, list_func_def_tk)
        all_full_tk.append(prompt_full_tokenizado)

    result = []
    print(
        "\n\n\033[1;36m[INFO]\033[0m Generating constrained responses...\n")
    for full_tk, prompt in zip(all_full_tk, list_prompt):
        answer = ai_model.ia_response(
            full_tk, prompt.get_prompt_tk(), list_func_def)
        result.append(answer)
        print("\n\033[1;32m[OK]\033[0m   Prompt processed.")
        dprint(f"    [DEBUG] Response: {answer}")
    rutas.write_output_json(result)

    print(
        "\n\n\n\033[1;32m[OK]\033[0m"
        f"   Results saved to: {rutas.output_path}")
    print("\n\n\033[1;35m[SUCCESS]\033[0m Execution completed successfully.\n")
