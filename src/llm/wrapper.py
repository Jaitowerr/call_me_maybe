from llm_sdk import Small_LLM_Model
from typing import List
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

