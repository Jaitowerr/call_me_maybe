from llm_sdk import Small_LLM_Model
from typing import List, Dict
from pydantic import BaseModel
import json
import sys
from src.object.Func_def import Func_def
from src.object.debug import Debug


class LLMWrapper():
    def __init__(self, model_name: str = 'Qwen/Qwen3-0.6B'):
        self.model = Small_LLM_Model(model_name)

    def encode_text(self, text: str) -> List[int]:
        """
        Esta es la función que pasaremos como 'encoder_func'.
        Convierte texto -> Tensor [[1, 2, 3]] -> Lista de IDs [1, 2, 3].
        Utiliza el SDK para el algoritmo BPE (complejidad O(N^2)), 
        pero la validación de la arquitectura de decodificación restringida
        se apoya exclusivamente en el vocabulario cargado en load_vocab().
        """
        tensor_ids = self.model.encode(text)  # [[1, 2, 3]]
        return tensor_ids.tolist()[0]  # [1, 2, 3]

    def decode_ids(self, ids: List[int]) -> str:
        """
        Convierte lista de IDs -> texto.
        Decodificación manual usando el vocabulario cargado.
        No depende del SDK.
        """
        text = "".join(self.id_to_tk_str.get(tid, "") for tid in ids)
        return text.replace("Ġ", " ").replace("Ċ", "\n")

    def load_vocab(self) -> None:
        """
        Carga el vocabulario desde la ruta que provee el SDK y construye
        id_to_tk_str y tk_str_to_id. Sale con sys.exit(1) si falla.
        """
        try:
            dprint = Debug().dprint
            vocab_path = self.model.get_path_to_vocab_file()
            dprint(f"\n    [DEBUG] Leyendo vocabulario desde: {vocab_path}")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)

            dprint(f"\n    [DEBUG] Tipo de raw: {type(raw)}")
            dprint(
                f"\n    [DEBUG] Ejemplo de contenido: {list(raw.items())[:3]}")

            if not isinstance(raw, dict):
                raise TypeError("El vocabulario no es un diccionario")

            # obtengo el string del token       Ejemplo: id_to_tk_str[12] -> "ĠHello"     Itera pares (k, v) = ("ĠHello", 12) y guarda {12: "ĠHello"}.
            self.id_to_tk_str = {int(v): k for k, v in raw.items()}
            # obtengo el id                     Ejemplo: tk_str_to_id["ĠHello"] -> 12     Guarda {"ĠHello": 12}.
            self.tk_str_to_id = {k: int(v) for k, v in raw.items()}

        except Exception as e:
            print(f"[ERROR] al cargar vocabulario: {e}")
            # import traceback
            # traceback.print_exc()
            sys.exit(1)

    # Con esto rpeguntamos directamente al modelo ¿Qué token viene???
    def get_logits(self, input_ids: list) -> list:
        """
        Devuelve los logits (lista de floats) para la secuencia input_ids.
        """
        try:
            logits = self.model.get_logits_from_input_ids(input_ids)
            return logits
        except Exception as e:
            print(f'[ERROR] al solicitar logits: {e}')
            sys.exit(1)

    def respuesta_ia(
        self,
        full_tk: List[int],
        prompt_tk: List[int],
        func_defs: List[Func_def] = None
    ) -> Dict:
        """
        Orquesta la generación completa con constrained decoding.
        Devuelve un dict con prompt, fn_name y args.
        """
        generated_ids: List[int] = self._generar_ids(
            full_tk, prompt_tk, func_defs)

        texto: str = self._ids_a_texto(generated_ids)
        # result: str = self._ids_a_texto(generated_ids)

        result: Dict = self._texto_a_dict(texto)

        return result

    def _ids_a_texto(self, ids: List[int]) -> str:
        """
        Convierte una lista de IDs generados por el modelo en texto.
        """
        return self.decode_ids(ids).strip()

    def _texto_a_dict(self, texto: str) -> Dict:
        """
        Parsea el texto JSON generado por el modelo -> dict.
        """
        try:
            return json.loads(texto)
        except json.JSONDecodeError as e:
            print(f"[ERROR] El modelo no generó un JSON válido: {e}")
            print(f"    [DEBUG] Texto recibido: {repr(texto)}")
            sys.exit(1)

    def _generar_ids(
            self,
            full_tk: List[int],
            prompt_tk: List[int],
            func_defs: List[Func_def] = None) -> List[int]:
        """
        Ejecuta una decodificación restringida (Constrained Decoding) mediante el uso
        de máscaras de logits para obligar al modelo a seguir un esquema JSON estricto.

        El método utiliza una máquina de estados para guiar la generación token a token:

        Estados de la generación:
            0: Inicio del JSON. Fuerza la secuencia exacta '{"prompt": "'.
            1: Contenido del Prompt. Inyecta los tokens del prompt original (escapando 
               comillas dobles) y cierra con '", '.
            2: Clave de Función. Fuerza la secuencia exacta '"fn_name": "'.
            3: Nombre de Función. Permite generación libre hasta detectar una comilla de 
               cierre. Identifica la función elegida para conocer sus parámetros.
            35/36: Transición. Asegura que tras la función exista un separador ', ' limpio.
            4: Inicio de Argumentos. Fuerza la secuencia exacta '"args": {'.
            5: Argumentos Dinámicos. Por cada parámetro de la función detectada:
               - Fuerza la clave exacta (ej: '"param": ').
               - Controla el tipo: si es string, fuerza comillas; si es número, 
                 permite dígitos.
               - Gestiona comas y espacios entre múltiples argumentos.

        El proceso finaliza automáticamente en cuanto el contador de llaves (brace_count) 
        llega a cero, garantizando un objeto JSON cerrado y evitando texto extra.
        """
        current_ids: List[int] = list(full_tk)
        generated_ids: List[int] = []
        generated_text: str = ""
        max_steps: int = 600
        brace_count: int = 0
        started: bool = False
        NEG_INF = float('-inf')
        dprint = Debug().dprint

        def get_id(s):
            return self.tk_str_to_id.get(s) or self.tk_str_to_id.get("Ġ" + s)


        raw_prompt = "".join(self.id_to_tk_str.get(t, "")
                             for t in prompt_tk).replace("Ġ", " ").replace("Ċ", "\n")

        safe_prompt = raw_prompt.replace('"', '\\"')

        prompt_tk = self.encode_text(safe_prompt)

        # --- Secuencias fijas ---
        header_tk = self.encode_text('{"prompt": "')
        prompt_close_tk = self.encode_text('", ')
        fn_key_tk = self.encode_text('"fn_name": "')
        args_key_tk = self.encode_text('"args": {')

        # --- Estado ---
        estado: int = 0
        idx: int = 0
        fn_buffer: str = ""
        func_detectada = None
        arg_keys: List[str] = []

        arg_idx: int = 0
        arg_sub_idx: int = 0
        arg_state: str = "key"
        value_quote_open: bool = False

        for step in range(max_steps):
            logits = self.get_logits(current_ids)
            mask = [NEG_INF] * len(logits)

            # ---------- MÁSCARA ----------
            if estado == 0:
                if idx < len(header_tk):
                    mask[header_tk[idx]] = 0.0
                    logits = mask

            elif estado == 1:
                total = len(prompt_tk) + len(prompt_close_tk)
                if idx < len(prompt_tk):
                    mask[prompt_tk[idx]] = 0.0
                    logits = mask
                elif idx < total:
                    mask[prompt_close_tk[idx - len(prompt_tk)]] = 0.0
                    logits = mask

            elif estado == 2:
                if idx < len(fn_key_tk):
                    mask[fn_key_tk[idx]] = 0.0
                    logits = mask

            elif estado == 3:
                pass  # libre

            elif estado == 35:
                if idx == 0:
                    cid = get_id(',')
                    if cid is not None:
                        mask[cid] = 0.0
                        logits = mask
                elif idx == 1:
                    sid = get_id(' ') or get_id('Ġ')
                    if sid is not None:
                        mask[sid] = 0.0
                        logits = mask

            elif estado == 36:
                sid = get_id(' ') or get_id('Ġ')
                if sid is not None:
                    mask[sid] = 0.0
                    logits = mask

            elif estado == 4:
                if idx < len(args_key_tk):
                    mask[args_key_tk[idx]] = 0.0
                    logits = mask

            elif estado == 5:
                if func_detectada and arg_idx < len(arg_keys):
                    current_key = arg_keys[arg_idx]
                    current_type = func_detectada.parameters[current_key]['type']

                    if arg_state == "key":
                        seq = self.encode_text(f'"{current_key}": ')
                        if arg_sub_idx < len(seq):
                            mask[seq[arg_sub_idx]] = 0.0
                            logits = mask

                    elif arg_state == "value":
                        if current_type == 'string':
                            if not value_quote_open:
                                q = get_id('"')
                                if q is not None:
                                    mask[q] = 0.0
                                    logits = mask
                        elif current_type == 'number':
                            pass  # libre total

                    elif arg_state == "sep":
                        if arg_idx < len(arg_keys) - 1:
                            if arg_sub_idx == 0:
                                cid = get_id(',')
                                if cid is not None:
                                    mask[cid] = 0.0
                                    logits = mask
                            elif arg_sub_idx == 1:
                                sid = get_id(' ') or get_id('Ġ')
                                if sid is not None:
                                    mask[sid] = 0.0
                                    logits = mask
                        else:
                            q = get_id('}')
                            if q is not None:
                                mask[q] = 0.0
                                logits = mask
                else:
                    q = get_id('}')
                    if q is not None:
                        mask[q] = 0.0
                        logits = mask

            # ---------- ELEGIR TOKEN ----------
            top_id = max(range(len(logits)), key=lambda i: logits[i])
            
            # INTERCEPTACIÓN: último argumento numérico no puede llevar coma
            if (estado == 5 and arg_state == "value" and func_detectada 
                    and arg_idx == len(arg_keys) - 1):
                current_key = arg_keys[arg_idx]
                if func_detectada.parameters[current_key]['type'] == 'number':
                    token_str_check = self.id_to_tk_str.get(top_id, "")
                    if ',' in token_str_check:
                        forced_id = get_id('}')
                        if forced_id is not None:
                            top_id = forced_id
            
            current_ids.append(top_id)
            generated_ids.append(top_id)
            token_str = self.id_to_tk_str.get(top_id, "<​UNK>")
            generated_text += token_str

            dprint(f"    [DEBUG] step={step:3d} estado={estado} idx={idx} "
                   f"arg_state={arg_state:4s} arg_idx={arg_idx}/{len(arg_keys)} "
                   f"tok={repr(token_str):16s} brace={brace_count} "
                   f"buf={repr(fn_buffer):30s}")

            # ---------- CONTADOR DE LLAVES ----------
            for char in token_str:
                if char == '{':
                    brace_count += 1
                    started = True
                elif char == '}':
                    brace_count -= 1
            if started and brace_count == 0:
                dprint(f"    [DEBUG] JSON completo en step {step}")
                break

            # ---------- TRANSICIONES ----------
            if estado == 0:
                idx += 1
                if idx >= len(header_tk):
                    estado, idx = 1, 0

            elif estado == 1:
                idx += 1
                if idx >= len(prompt_tk) + len(prompt_close_tk):
                    estado, idx = 2, 0

            elif estado == 2:
                idx += 1
                if idx >= len(fn_key_tk):
                    estado, idx, fn_buffer = 3, 0, ""

            elif estado == 3:
                if '"' in token_str:
                    partes = token_str.split('"', 1)
                    fn_buffer += partes[0]
                    clean = fn_buffer.strip()
                    dprint(f"    [DEBUG] Nombre detectado: {repr(clean)}")
                    if func_defs:
                        for fd in func_defs:
                            if fd.name == clean:
                                func_detectada = fd
                                arg_keys = list(fd.parameters.keys())
                                dprint(
                                    f"    [DEBUG] Funcion OK: {fd.name} args={arg_keys}")
                                break
                    resto = partes[1] if len(partes) > 1 else ""
                    if resto.startswith(', ') or resto.startswith(',Ġ'):
                        estado, idx = 4, 0
                    elif resto.startswith(','):
                        estado, idx = 36, 0
                    else:
                        estado, idx = 35, 0
                else:
                    fn_buffer += token_str

            elif estado == 35:
                idx += 1
                if idx >= 2:
                    estado, idx = 4, 0

            elif estado == 36:
                estado, idx = 4, 0

            elif estado == 4:
                idx += 1
                if idx >= len(args_key_tk):
                    estado, idx, arg_idx, arg_sub_idx, arg_state, value_quote_open = 5, 0, 0, 0, "key", False

            elif estado == 5:
                if func_detectada and arg_idx < len(arg_keys):
                    current_key = arg_keys[arg_idx]
                    current_type = func_detectada.parameters[current_key]['type']

                    if arg_state == "key":
                        seq = self.encode_text(f'"{current_key}": ')
                        arg_sub_idx += 1
                        if arg_sub_idx >= len(seq):
                            arg_state, arg_sub_idx, value_quote_open = "value", 0, False

                    elif arg_state == "value":
                        if current_type == 'string':
                            if not value_quote_open:
                                value_quote_open = True
                            else:
                                if '"' in token_str:
                                    after = token_str.rsplit('"', 1)[-1]
                                    if ',' in after:
                                        arg_idx += 1
                                        arg_state, arg_sub_idx = "key", 0
                                    else:
                                        arg_state, arg_sub_idx = "sep", 0
                        elif current_type == 'number':
                            if ',' in token_str:
                                if arg_idx < len(arg_keys) - 1:
                                    arg_idx += 1
                                    arg_state, arg_sub_idx = "key", 0
                                else:
                                    # Último argumento: ignorar coma, forzar cierre
                                    arg_idx = len(arg_keys)
                            elif '}' in token_str:
                                arg_idx = len(arg_keys)

                    elif arg_state == "sep":
                        if arg_idx < len(arg_keys) - 1:
                            arg_sub_idx += 1
                            if arg_sub_idx >= 2:
                                arg_idx += 1
                                arg_state, arg_sub_idx = "key", 0
                        else:
                            arg_idx = len(arg_keys)

        return generated_ids

    # def _generar_ids(
    #         self,
    #         full_tk: List[int],
    #         prompt_tk: List[int],
    #         fn_names_tk: List[List[int]]) -> List[int]:
    #     """
    #     Genera una secuencia de tokens de respuesta mediante un bucle token a token.

    #     En cada iteración:
    #     1. Se obtienen los logits del modelo dado el contexto acumulado (current_ids).
    #     2. Se aplica una máscara de logits (constrained decoding) para restringir
    #         los tokens válidos según el estado actual de la generación.
    #         - Estado inicial (not started): solo se permite el token '{'.
    #     3. Se selecciona el token con mayor puntuación (argmax).
    #     4. Se actualiza el contexto y se registra el token generado.
    #     5. Se controla la profundidad de llaves (brace brace_count counter) para
    #         detectar cuándo el objeto JSON está completo.

    #     El bucle termina cuando todas las llaves abiertas han sido cerradas
    #     (brace_count == 0 y started == True), o cuando se alcanza el límite de pasos.

    #     Args:
    #         full_tk: Lista de tokens que representa el contexto completo de entrada
    #                 (instrucciones + funciones disponibles + query del usuario).
    #         prompt_tk: Lista de tokens del prompt original del usuario.
    #                 Reservado para uso en fases posteriores del constrained decoding.

    #     Returns:
    #         Lista de enteros con los IDs de los tokens generados por el modelo,
    #         sin incluir el contexto de entrada (full_tk).
    #     """
    #     current_ids: List[int] = list(full_tk)
    #     generated_ids: List[int] = []
    #     generated_text: str = ""
    #     max_steps: int = 200
    #     brace_count: int = 0
    #     started: bool = False
    #     NEG_INF = float('-inf')

    #     for step in range(max_steps):
    #         logits = self.get_logits(current_ids)

    #         if not started:
    #             id_permitido = self.tk_str_to_id.get('{') or self.tk_str_to_id.get('Ġ{')

    #             if id_permitido is not None:
    #                 mask = [NEG_INF] * len(logits)
    #                 mask[id_permitido] = 0.0
    #                 logits = mask

    #         top_id = max(range(len(logits)), key=lambda i: logits[i])

    #         current_ids.append(top_id)
    #         generated_ids.append(top_id)

    #         token_str = self.id_to_tk_str.get(top_id, "<UNK>")
    #         generated_text += token_str

    #         for char in token_str:
    #             if char == '{':
    #                 brace_count += 1
    #                 started = True
    #             elif char == '}':
    #                 brace_count -= 1

    #         if started and brace_count == 0:
    #             break

    #     return generated_ids
