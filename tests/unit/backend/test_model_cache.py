"""Unit tests for ModelCache and get_model_cache."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.backend.model_cache import ModelCache, get_model_cache, _MODEL_CACHE


class TestModelCacheInit:
    """Test ModelCache initialization."""

    @pytest.mark.unit
    def test_init_default_max_models(self):
        """Test initialization with default max_models."""
        cache = ModelCache()
        assert cache.max_models == 2
        assert len(cache._cache) == 0

    @pytest.mark.unit
    def test_init_custom_max_models(self):
        """Test initialization with custom max_models."""
        cache = ModelCache(max_models=5)
        assert cache.max_models == 5

    @pytest.mark.unit
    def test_init_single_model_cache(self):
        """Test initialization with max_models=1."""
        cache = ModelCache(max_models=1)
        assert cache.max_models == 1

    @pytest.mark.unit
    def test_init_large_cache(self):
        """Test initialization with large max_models."""
        cache = ModelCache(max_models=100)
        assert cache.max_models == 100


class TestMakeKey:
    """Test ModelCache._make_key method."""

    @pytest.mark.unit
    def test_make_key_with_none_lm_path(self):
        """Test key generation with None lm_path."""
        cache = ModelCache()
        key = cache._make_key("whisper", "tiny", None)
        assert key == ("whisper", "tiny", None)

    @pytest.mark.unit
    def test_make_key_with_lm_path(self):
        """Test key generation with lm_path."""
        cache = ModelCache()
        key = cache._make_key("wav2vec2", "xlsr-53", "/path/to/lm.arpa")
        assert key == ("wav2vec2", "xlsr-53", "/path/to/lm.arpa")

    @pytest.mark.unit
    def test_make_key_consistency(self):
        """Test that _make_key produces consistent keys."""
        cache = ModelCache()
        key1 = cache._make_key("mms", "1b-all", None)
        key2 = cache._make_key("mms", "1b-all", None)
        assert key1 == key2

    @pytest.mark.unit
    def test_make_key_distinguishes_models(self):
        """Test that different models produce different keys."""
        cache = ModelCache()
        key1 = cache._make_key("whisper", "base", None)
        key2 = cache._make_key("whisper", "large", None)
        assert key1 != key2

    @pytest.mark.unit
    def test_make_key_distinguishes_lm_paths(self):
        """Test that different lm_paths produce different keys."""
        cache = ModelCache()
        key1 = cache._make_key("wav2vec2", "xlsr", "/lm1.arpa")
        key2 = cache._make_key("wav2vec2", "xlsr", "/lm2.arpa")
        assert key1 != key2


class TestCacheGet:
    """Test ModelCache.get method."""

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_get_cache_miss_loads_new_evaluator(self, mock_evaluator_class):
        """Test that cache miss loads a new evaluator."""
        mock_instance = Mock()
        mock_evaluator_class.return_value = mock_instance
        
        cache = ModelCache()
        result = cache.get("whisper", "tiny", None)

        # Should create evaluator and call load_model
        mock_evaluator_class.assert_called_once_with(
            model_type="whisper",
            model_name="tiny",
            lm_path=None
        )
        mock_instance.load_model.assert_called_once()
        assert result == mock_instance

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_get_cache_hit_returns_cached(self, mock_evaluator_class):
        """Test that cache hit returns cached evaluator."""
        mock_instance = Mock()
        mock_evaluator_class.return_value = mock_instance
        
        cache = ModelCache()
        
        # First call: cache miss
        result1 = cache.get("whisper", "tiny", None)
        
        # Second call: cache hit
        result2 = cache.get("whisper", "tiny", None)

        # Should only create one evaluator
        assert mock_evaluator_class.call_count == 1
        assert result1 is result2

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_get_cache_hit_updates_lru_order(self, mock_evaluator_class):
        """Test that cache hit moves item to end (most recently used)."""
        mock_instances = [Mock() for _ in range(3)]
        mock_evaluator_class.side_effect = mock_instances
        
        cache = ModelCache(max_models=2)
        
        # Load two models
        cache.get("whisper", "tiny", None)
        cache.get("whisper", "base", None)
        
        # Access first model again (should move to end)
        cache.get("whisper", "tiny", None)
        
        # Load a third model (should evict "base" not "tiny")
        cache.get("whisper", "large", None)
        
        # "base" should be evicted, "tiny" and "large" should be cached
        keys = list(cache._cache.keys())
        assert ("whisper", "tiny", None) in keys
        assert ("whisper", "large", None) in keys
        assert ("whisper", "base", None) not in keys

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_get_lru_eviction_at_capacity(self, mock_evaluator_class):
        """Test LRU eviction when cache reaches capacity."""
        mock_instances = [Mock() for _ in range(3)]
        mock_evaluator_class.side_effect = mock_instances
        
        cache = ModelCache(max_models=2)
        
        # Load models to reach capacity
        cache.get("whisper", "tiny", None)
        cache.get("whisper", "base", None)
        assert len(cache._cache) == 2
        
        # Load third model: should evict least recently used (tiny)
        cache.get("whisper", "large", None)
        
        assert len(cache._cache) == 2
        keys = list(cache._cache.keys())
        assert ("whisper", "tiny", None) not in keys
        assert ("whisper", "base", None) in keys
        assert ("whisper", "large", None) in keys

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_get_different_lm_paths_different_cache_entries(self, mock_evaluator_class):
        """Test that different lm_paths create separate cache entries."""
        mock_instances = [Mock() for _ in range(2)]
        mock_evaluator_class.side_effect = mock_instances
        
        cache = ModelCache()
        
        result1 = cache.get("wav2vec2", "xlsr", "/lm1.arpa")
        result2 = cache.get("wav2vec2", "xlsr", "/lm2.arpa")
        
        assert result1 is not result2
        assert len(cache._cache) == 2
        assert mock_evaluator_class.call_count == 2

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_get_different_model_types_different_cache_entries(self, mock_evaluator_class):
        """Test that different model types create separate cache entries."""
        mock_instances = [Mock() for _ in range(2)]
        mock_evaluator_class.side_effect = mock_instances
        
        cache = ModelCache()
        
        result1 = cache.get("whisper", "tiny", None)
        result2 = cache.get("wav2vec2", "tiny", None)
        
        assert result1 is not result2
        assert len(cache._cache) == 2
        assert mock_evaluator_class.call_count == 2

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_get_multiple_lru_evictions(self, mock_evaluator_class):
        """Test multiple LRU evictions in sequence."""
        mock_instances = [Mock() for _ in range(5)]
        mock_evaluator_class.side_effect = mock_instances
        
        cache = ModelCache(max_models=2)
        
        cache.get("whisper", "tiny", None)
        cache.get("whisper", "base", None)
        cache.get("whisper", "small", None)  # Evicts tiny
        cache.get("whisper", "medium", None)  # Evicts base
        
        keys = list(cache._cache.keys())
        assert ("whisper", "tiny", None) not in keys
        assert ("whisper", "base", None) not in keys
        assert ("whisper", "small", None) in keys
        assert ("whisper", "medium", None) in keys

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_get_cache_size_never_exceeds_max(self, mock_evaluator_class):
        """Test that cache size never exceeds max_models."""
        mock_instances = [Mock() for _ in range(10)]
        mock_evaluator_class.side_effect = mock_instances
        
        cache = ModelCache(max_models=3)
        
        for i in range(10):
            cache.get("whisper", f"model_{i}", None)
            assert len(cache._cache) <= 3


class TestCacheClear:
    """Test ModelCache.clear method."""

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_clear_empty_cache(self, mock_evaluator_class):
        """Test clearing an empty cache."""
        cache = ModelCache()
        cache.clear()
        assert len(cache._cache) == 0

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_clear_nonempty_cache(self, mock_evaluator_class):
        """Test clearing a cache with entries."""
        mock_instance = Mock()
        mock_evaluator_class.return_value = mock_instance
        
        cache = ModelCache()
        cache.get("whisper", "tiny", None)
        cache.get("whisper", "base", None)
        
        assert len(cache._cache) == 2
        cache.clear()
        assert len(cache._cache) == 0

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_clear_and_refill(self, mock_evaluator_class):
        """Test that cache can be refilled after clearing."""
        mock_instance = Mock()
        mock_evaluator_class.return_value = mock_instance
        
        cache = ModelCache()
        cache.get("whisper", "tiny", None)
        cache.clear()
        
        # Should create new evaluator (not return cached)
        result = cache.get("whisper", "tiny", None)
        assert result == mock_instance


class TestCacheInfo:
    """Test ModelCache.info method."""

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_info_empty_cache(self, mock_evaluator_class):
        """Test info() on empty cache."""
        cache = ModelCache(max_models=5)
        info = cache.info()
        
        assert info["max_models"] == 5
        assert info["size"] == 0
        assert info["entries"] == []

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_info_single_entry(self, mock_evaluator_class):
        """Test info() with single cache entry."""
        mock_instance = Mock()
        mock_evaluator_class.return_value = mock_instance
        
        cache = ModelCache()
        cache.get("whisper", "tiny", None)
        
        info = cache.info()
        assert info["max_models"] == 2
        assert info["size"] == 1
        assert len(info["entries"]) == 1
        assert info["entries"][0]["model_type"] == "whisper"
        assert info["entries"][0]["model_name"] == "tiny"
        assert info["entries"][0]["lm_path"] is None

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_info_multiple_entries(self, mock_evaluator_class):
        """Test info() with multiple cache entries."""
        mock_instance = Mock()
        mock_evaluator_class.return_value = mock_instance
        
        cache = ModelCache()
        cache.get("whisper", "tiny", None)
        cache.get("wav2vec2", "xlsr", "/path/to/lm.arpa")
        
        info = cache.info()
        assert info["size"] == 2
        assert len(info["entries"]) == 2
        
        # Check first entry
        assert info["entries"][0]["model_type"] == "whisper"
        assert info["entries"][0]["model_name"] == "tiny"
        
        # Check second entry
        assert info["entries"][1]["model_type"] == "wav2vec2"
        assert info["entries"][1]["lm_path"] == "/path/to/lm.arpa"

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_info_entry_structure(self, mock_evaluator_class):
        """Test that info() entries have correct structure."""
        mock_instance = Mock()
        mock_evaluator_class.return_value = mock_instance
        
        cache = ModelCache()
        cache.get("mms", "1b-all", "/lm.arpa")
        
        info = cache.info()
        entry = info["entries"][0]
        
        assert "model_type" in entry
        assert "model_name" in entry
        assert "lm_path" in entry
        assert entry["model_type"] == "mms"
        assert entry["model_name"] == "1b-all"
        assert entry["lm_path"] == "/lm.arpa"


class TestGetModelCacheSingleton:
    """Test get_model_cache singleton function."""

    @pytest.mark.unit
    def test_get_model_cache_returns_cache_instance(self):
        """Test that get_model_cache returns a ModelCache instance."""
        # Reset global cache
        import src.backend.model_cache as mc
        mc._MODEL_CACHE = None
        
        cache = get_model_cache()
        assert isinstance(cache, ModelCache)

    @pytest.mark.unit
    def test_get_model_cache_returns_same_instance(self):
        """Test that get_model_cache returns the same singleton."""
        # Reset global cache
        import src.backend.model_cache as mc
        mc._MODEL_CACHE = None
        
        cache1 = get_model_cache()
        cache2 = get_model_cache()
        
        assert cache1 is cache2

    @pytest.mark.unit
    @patch("src.backend.model_cache.ASREvaluator")
    def test_get_model_cache_singleton_persistence(self, mock_evaluator_class):
        """Test that cache persists across calls to get_model_cache."""
        # Reset global cache
        import src.backend.model_cache as mc
        mc._MODEL_CACHE = None
        
        mock_instance = Mock()
        mock_evaluator_class.return_value = mock_instance
        
        # Load a model through the singleton
        cache1 = get_model_cache()
        cache1.get("whisper", "tiny", None)
        
        # Get the singleton again
        cache2 = get_model_cache()
        
        # Should still have the cached model
        assert len(cache2._cache) == 1
        assert ("whisper", "tiny", None) in cache2._cache

    @pytest.mark.unit
    def test_get_model_cache_default_capacity(self):
        """Test that singleton has default capacity."""
        # Reset global cache
        import src.backend.model_cache as mc
        mc._MODEL_CACHE = None
        
        cache = get_model_cache()
        assert cache.max_models == 2
