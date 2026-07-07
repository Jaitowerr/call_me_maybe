from llm_sdk import Small_LLM_Model
from typing import List, Dict
from pydantic import BaseModel
import json
import sys

class LLMWrapper():
    def __init__(self, model_name: str = 'Qwen/Qwen3-0.6B'):
        self.model = Small_LLM_Model(model_name)
    
    def encode_text(self, text:str) -> List[int]:
        """
        Esta es la función que pasaremos como 'encoder_func'.
        Convierte texto -> Tensor [[1, 2, 3]] -> Lista de IDs [1, 2, 3].
        """
        tensor_ids = self.model.encode(text) #[[1, 2, 3]]
        return tensor_ids.tolist()[0] #[1, 2, 3]

    def decode_ids(self, ids: List[int]) -> str:
        """Convierte lista de IDs -> texto."""
        return self.model.decode(ids)

    def load_vocab(self) -> None:
        """
        Carga el vocabulario desde la ruta que provee el SDK y construye
        id_to_tk_str y tk_str_to_id. Sale con sys.exit(1) si falla.
        """
        try:
            vocab_path = self.model.get_path_to_vocab_file()
            print(f"[DEBUG] Leyendo vocabulario desde: {vocab_path}")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            
            print(f"[DEBUG] Tipo de raw: {type(raw)}")
            print(f"[DEBUG] Ejemplo de contenido: {list(raw.items())[:3]}")

            if not isinstance(raw, dict):
                raise TypeError("El vocabulario no es un diccionario")

            self.id_to_tk_str = {int(v): k for k, v in raw.items()}		#obtengo el string del token       Ejemplo: id_to_tk_str[12] -> "ĠHello"     Itera pares (k, v) = ("ĠHello", 12) y guarda {12: "ĠHello"}.
            self.tk_str_to_id = {k: int(v) for k, v in raw.items()}		#obtengo el id                     Ejemplo: tk_str_to_id["ĠHello"] -> 12     Guarda {"ĠHello": 12}.

        except Exception as e:
            print(f"[ERROR] al cargar vocabulario: {e}")
            # import traceback
            # traceback.print_exc()
            sys.exit(1)

    def get_logits(self, input_ids: list) -> list:					#Con esto rpeguntamos directamente al modelo ¿Qué token viene???
        """
        Devuelve los logits (lista de floats) para la secuencia input_ids.
        """
        try:
            logits = self.model.get_logits_from_input_ids(input_ids)
            return logits
        except Exception as e:
            print(f'[ERROR] al solicitar logits: {e}')
            sys.exit(1)

    def respuesta_ia(self, full_tk: List[int], prompt_tk: List[int]) -> Dict:
        """
        Orquesta la generación completa con constrained decoding.
        Devuelve un dict con prompt, fn_name y args.
        """
        generated_ids: List[int] = self._generar_ids(full_tk, prompt_tk)

        texto: str = self._ids_a_texto(generated_ids)
        # result: str = self._ids_a_texto(generated_ids)

        result: Dict = self._texto_a_dict(texto)

        return result


    def _ids_a_texto(self, ids: List[int]) -> str:
        texto = self.decode_ids(ids)
        # Extraer solo lo que hay entre <json> y </json>
        if "<json>" in texto and "</json>" in texto:
            inicio = texto.index("<json>") + len("<json>")
            fin = texto.index("</json>")
            return texto[inicio:fin].strip()
        # Si no hay etiquetas, devolver tal cual (fallback)
        return texto.strip()


    def _texto_a_dict(self, texto: str) -> Dict:
        """
        Parsea el texto JSON generado por el modelo -> dict.
        """
        try:
            return json.loads(texto)
        except json.JSONDecodeError as e:
            print(f"[ERROR] El modelo no generó un JSON válido: {e}")
            print(f"[DEBUG] Texto recibido: {repr(texto)}")
            sys.exit(1)

    def _generar_ids(self, full_tk: List[int], prompt_tk: List[int]) -> List[int]:
        """
        Bucle token a token sin restricciones (por ahora).
        Devuelve solo los ids generados, sin el contexto inicial.
        """
        current_ids: List[int] = list(full_tk)
        generated_ids: List[int] = []
        generated_text: str = ""
        max_steps: int = 200
        depth: int = 0          # contador de llaves abiertas
        started: bool = False   # hemos visto al menos una '{'

        for step in range(max_steps):
            logits = self.get_logits(current_ids)

            top_id = max(range(len(logits)), key=lambda i: logits[i])

            current_ids.append(top_id)
            generated_ids.append(top_id)

            token_str = self.id_to_tk_str.get(top_id, "<UNK>")
            generated_text += token_str

            # actualizar contador de llaves
            for char in token_str:
                if char == '{':
                    depth += 1
                    started = True
                elif char == '}':
                    depth -= 1

            # si hemos empezado y todas las llaves están cerradas -> JSON completo
            if started and depth == 0:
                break
        else:
            print("[WARN] Límite de pasos alcanzado sin JSON completo.")

        return generated_ids


    # def _generar_ids(self, full_tk: List[int], prompt_tk: List[int]) -> List[int]:
    #     current_ids = list(full_tk)
    #     generated_ids = []
    #     NEG_INF = float('-inf')

    #     # --- ESTADO 0: Obligar a empezar con '{' ---
    #     # 1. Pedimos las puntuaciones (logits) a la IA basándonos en el contexto
    #     logits = self.get_logits(current_ids)
        
    #     # 2. Buscamos el ID de la llave en nuestro vocabulario
    #     id_llave = self.tk_str_to_id.get('{') or self.tk_str_to_id.get('Ġ{')

    #     # 3. MÁSCARA: Creamos una nueva lista de logits donde TODO es -inf
    #     # excepto la posición de nuestra llave, que mantiene su puntuación original.
    #     logits_restringidos = [NEG_INF] * len(logits)
    #     if id_llave is not None:
    #         logits_restringidos[id_llave] = logits[id_llave]
    #     else:
    #         # Si por algún motivo no existe la llave en el vocabulario (raro)
    #         print("[ERROR] No se encontró el token de llave '{'")
    #         sys.exit(1)

    #     # 4. Ahora la IA elige el máximo de la lista restringida
    #     # argmax: busca el índice del valor más alto
    #     token_elegido = max(range(len(logits_restringidos)), key=lambda i: logits_restringidos[i])

    #     # 5. Guardamos lo que la IA "ha elegido" bajo nuestra presión
    #     current_ids.append(token_elegido)
    #     generated_ids.append(token_elegido)
        
    #     # Imprimimos para ver que funciona
    #     print(f"[DEBUG] Generado primer token: {self.decode_ids([token_elegido])}")

    #     return generated_ids