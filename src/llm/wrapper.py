from llm_sdk.llm_sdk import Small_LLM_Model
from typing import List
from pydantic import BaseModel

class LLMWrapper(BaseModel):
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











