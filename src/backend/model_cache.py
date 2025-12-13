"""LRU cache for ASR evaluators.

This cache keeps a small number of loaded `ASREvaluator` instances in memory
to avoid repeated model loading overhead. Use `get_model_cache()` to access the
singleton instance.
"""

from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any

from src.evaluation.evaluator import ASREvaluator


class ModelCache:
	"""LRU cache for loaded ASR models."""

	def __init__(self, max_models: int = 2) -> None:
		self.max_models = max_models
		self._cache: "OrderedDict[Tuple[str, str, Optional[str]], ASREvaluator]" = OrderedDict()

	def _make_key(self, model_type: str, model_name: str, lm_path: Optional[str]) -> Tuple[str, str, Optional[str]]:
		return (model_type, model_name, lm_path)

	def get(self, model_type: str, model_name: str, lm_path: Optional[str] = None) -> ASREvaluator:
		"""Return a cached evaluator or load and cache a new one."""
		key = self._make_key(model_type, model_name, lm_path)

		if key in self._cache:
			# Move to end to mark as recently used
			self._cache.move_to_end(key)
			return self._cache[key]

		# Cache miss: load new evaluator
		evaluator = ASREvaluator(model_type=model_type, model_name=model_name, lm_path=lm_path)
		evaluator.load_model()

		# Evict least recently used if at capacity
		if len(self._cache) >= self.max_models:
			self._cache.popitem(last=False)

		self._cache[key] = evaluator
		return evaluator

	def clear(self) -> None:
		"""Remove all cached evaluators."""
		self._cache.clear()

	def info(self) -> Dict[str, Any]:
		"""Return basic cache diagnostics."""
		entries = []
		for (model_type, model_name, lm_path) in self._cache.keys():
			entries.append({
				"model_type": model_type,
				"model_name": model_name,
				"lm_path": lm_path
			})

		return {
			"max_models": self.max_models,
			"size": len(self._cache),
			"entries": entries,
		}


_MODEL_CACHE: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
	"""Return the singleton model cache instance."""
	global _MODEL_CACHE
	if _MODEL_CACHE is None:
		_MODEL_CACHE = ModelCache()
	return _MODEL_CACHE

