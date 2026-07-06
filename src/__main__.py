import sys
from src.object.Parse import Config
from src.object.Prompt_io import Prompt_io
from src.object.Func_def import Func_def
from pathlib import Path
from src.llm.wrapper import LLMWrapper
from src.llm.prompt_builder import PromptBuilder
from typing import List


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
        # print(f"  [{i}] prompt = {p.prompt!r}  fn_name={p.fn_name} args={p.args}")
        print(f"  [{i}] prompt = {p.prompt!r}")
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

    print('Creamos un str de cada funcion para tokeizar luego\n')
    ai_model = LLMWrapper()
    
    for _ in list_prompt: _.tokenize(ai_model.encode_text)
    print('---> tokeniazación str prompt Prompt_io')
    for _ in list_func_def: _.tokenize_signature(ai_model.encode_text)
    print('---> tokeniazación str de Func_def, funciones tokenizadas')
    
    print("\n\n\n\n\n\n=== COMPROBANDO PROMPT BUILDER ===")
    builder_tk = PromptBuilder(ai_model.encode_text)

    for prompt in list_prompt:
        full_prompt_tk = builder_tk.build_tk(prompt, list_func_def)
        print(f"✅ Tokens generados: {len(full_prompt_tk)}")
        print(f"    - PromptBuilder construyó correctamente el prompt : {prompt.prompt}")
    

    print("\n\n\n\n\n=== PRUEBA DE MODELO: VOCABULARIO Y LOGITS ===")
    
    ai_model.load_vocab()    # Cargar vocabulario una vez
    print(f"✅ Vocabulario cargado. Total tokens: {len(ai_model.id_to_tk_str)}\n")

    #Creamos lsita de funciones tokenizadas, la extraemos
    list_func_def_tk: List[int] = []
    for f in list_func_def:
        list_func_def_tk.extend(f.get_signature_tk())
    
    # Bucle por cada prompt construido ya con el build_tk
    all_full_tk: List[List[int]] = []
    for prompt in list_prompt:
        prompt_full_tokenizado = builder_tk.build_tk(prompt, list_func_def_tk)
        all_full_tk.append(prompt_full_tokenizado)

    result = []
    print("\n\n\n\n\n=== PRUEBA DE GENERACIÓN TOKEN-A-TOKEN (SIN RESTRICCIONES) ===")
    for full_tk, prompt in zip(all_full_tk, list_prompt):
        answer = ai_model.respuesta_ia(full_tk, prompt.get_prompt_tk())
        result.append(answer)
        break
    rutas.write_output_json(result)


    
    print('---------------------------------------------------> fiiiin')



# def prueba_completa_vocabulario_y_logits(ai_model, full_tk) -> dict:
#     """
#     Prueba de generación token-a-token sin restricciones.
#     Recibe el contexto ya preparado (full_tk) y genera hasta encontrar '}'.
#     """
#     try:
#         ai_model.load_vocab()
#         print(f"✅ Vocabulario cargado. Total tokens: {len(ai_model.id_to_tk_str)}\n")
#     except Exception:
#         print('ERROR al cargar vocabulario')
#         return

#     # Copia del contexto que iremos extendiendo
#     current_ids = list(full_tk)  # evita mutar el original accidentalmente

#     generated_ids = []   # solo los ids generados (no incluye el contexto)
#     generated_text = ""  # acumulador para ver el texto a nivel token (concatenando id_to_tk_str)

#     max_steps = 200               # tope para evitar loops infinitos
#     stop_when_seen_closing_brace = True

#     for step in range(max_steps):
#         logits = ai_model.get_logits(current_ids)

#         # elegir el id con mayor logit (argmax)
#         top_id = max(range(len(logits)), key=lambda i: logits[i])

#         # añadirlo a las secuencias
#         current_ids.append(top_id)
#         generated_ids.append(top_id)

#         # convertir el token a string usando el vocab
#         token_str = ai_model.id_to_tk_str.get(top_id, "<​UNK>")
#         generated_text += token_str

#         # imprimir progreso por pasos para debug
#         print(f"[{step+1:03d}] id={top_id} token={repr(token_str)}  --> texto_parcial: {repr(generated_text)}")

#         # condición simple de parada: si generó '}'
#         if stop_when_seen_closing_brace and ("}" in generated_text):
#             print("\n✅ Detectado '}' en la salida generada. Parando la prueba.")
#             break
#     else:
#         print("\n⚠️ Límite de pasos alcanzado sin detectar '}'.")

#     # Mostrar resultado final "detokenizado"
#     try:
#         final_text = ai_model.decode_ids(current_ids)  # si tu wrapper tiene decode_ids
#     except Exception:
#         # fallback: reconstruir a partir del vocab (menos preciso, pero útil)
#         final_text = "".join(ai_model.id_to_tk_str.get(i, "") for i in current_ids)

#     print("\n================ RESULTADO FINAL ================")

#     print("\n--- SOLO LO GENERADO POR EL MODELO ---")
#     print(generated_text)

#     print("\n--- TEXTO COMPLETO (PROMPT + RESPUESTA) ---")
#     print(final_text)

#     print("\n--- IDS GENERADOS ---")
#     print(generated_ids)

#     print("\n--- TOKENS GENERADOS ---")
#     for token_id in generated_ids:
#         print(f"{token_id:>6} -> {repr(ai_model.id_to_tk_str.get(token_id, '<UNK>'))}")

#     print("\n================================================")
