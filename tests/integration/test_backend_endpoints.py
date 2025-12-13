"""Integration tests for FastAPI backend endpoints."""
import json
import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def client():
    """Create FastAPI test client."""
    # Import only after mocking whisper
    from src.backend.endpoints import router as api_router

    app = FastAPI()
    app.include_router(api_router, prefix="/api")

    return TestClient(app)


@pytest.fixture
def fake_cache(monkeypatch):
    """Mock model cache for testing."""
    from src.backend import endpoints
    cache = Mock()
    monkeypatch.setattr(endpoints, "get_model_cache", lambda: cache)
    return cache


class TestEvaluateEndpoint:
    """Test /evaluate endpoint integration."""

    @pytest.mark.integration
    def test_evaluate_endpoint_success(self, client, fake_cache):
        """Test successful evaluation request."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 10,
            "failed_samples": 0,
            "overall_wer": 30.0,
            "overall_cer": 15.0,
            "overall_bleu": 65.0,
            "per_dialect_wer": {"BE": 28.0},
            "per_dialect_cer": {"BE": 14.0},
            "per_dialect_bleu": {"BE": 67.0}
        }
        fake_cache.get.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={
                "model_type": "whisper",
                "model": "tiny",
                "limit": 10
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "tiny"
        assert data["total_samples"] == 10
        assert data["overall_wer"] == 30.0

    @pytest.mark.integration
    def test_evaluate_endpoint_validation_error(self, client):
        """Test endpoint with invalid request body."""
        response = client.post(
            "/api/evaluate",
            json={"invalid": "data"}
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.integration
    def test_evaluate_endpoint_missing_model(self, client):
        """Test endpoint with missing model field."""
        response = client.post(
            "/api/evaluate",
            json={"model_type": "whisper"}
        )

        assert response.status_code == 422

    @pytest.mark.integration
    def test_evaluate_endpoint_with_limit(self, client, fake_cache):
        """Test evaluation request with sample limit."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 5,
            "failed_samples": 0,
            "overall_wer": 25.0,
            "overall_cer": 12.0,
            "overall_bleu": 70.0,
            "per_dialect_wer": {},
            "per_dialect_cer": {},
            "per_dialect_bleu": {}
        }
        fake_cache.get.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={
                "model_type": "whisper",
                "model": "base",
                "limit": 5
            }
        )

        assert response.status_code == 200
        mock_evaluator.evaluate_dataset.assert_called_once()
        _, kwargs = mock_evaluator.evaluate_dataset.call_args
        assert kwargs.get("limit") == 5

    @pytest.mark.integration
    def test_evaluate_endpoint_wav2vec2_model(self, client, fake_cache):
        """Test evaluation with Wav2Vec2 model type."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 10,
            "failed_samples": 1,
            "overall_wer": 35.0,
            "overall_cer": 18.0,
            "overall_bleu": 60.0,
            "per_dialect_wer": {"BE": 33.0, "ZH": 37.0},
            "per_dialect_cer": {"BE": 16.0, "ZH": 20.0},
            "per_dialect_bleu": {"BE": 62.0, "ZH": 58.0}
        }
        fake_cache.get.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={
                "model_type": "wav2vec2",
                "model": "facebook/wav2vec2-large-xlsr-53-german"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "facebook/wav2vec2-large-xlsr-53-german"
        assert data["failed_samples"] == 1

    @pytest.mark.integration
    def test_evaluate_endpoint_mms_model(self, client, fake_cache):
        """Test evaluation with MMS model type."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 20,
            "failed_samples": 2,
            "overall_wer": 28.0,
            "overall_cer": 12.0,
            "overall_bleu": 72.0,
            "per_dialect_wer": {"BE": 25.0},
            "per_dialect_cer": {"BE": 10.0},
            "per_dialect_bleu": {"BE": 75.0}
        }
        fake_cache.get.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={
                "model_type": "mms",
                "model": "facebook/mms-1b-all"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "facebook/mms-1b-all"

    @pytest.mark.integration
    def test_evaluate_endpoint_handles_error(self, client, fake_cache):
        """Test endpoint handles evaluation errors gracefully."""
        fake_cache.get.side_effect = Exception("Model loading failed")

        response = client.post(
            "/api/evaluate",
            json={
                "model_type": "whisper",
                "model": "nonexistent"
            }
        )

        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()

    @pytest.mark.integration
    def test_evaluate_endpoint_invalid_model_type(self, client):
        """Test endpoint rejects invalid model type."""
        response = client.post(
            "/api/evaluate",
            json={
                "model_type": "invalid_type",
                "model": "test"
            }
        )

        assert response.status_code == 422

    @pytest.mark.integration
    def test_evaluate_endpoint_negative_limit(self, client):
        """Test endpoint rejects negative limit."""
        response = client.post(
            "/api/evaluate",
            json={
                "model_type": "whisper",
                "model": "tiny",
                "limit": -5
            }
        )

        assert response.status_code == 422


class TestEndpointResponseFormat:
    """Test endpoint response format and structure."""

    @pytest.mark.integration
    def test_response_contains_all_metrics(self, client, fake_cache):
        """Test response contains all required metrics."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 50,
            "failed_samples": 5,
            "overall_wer": 32.5,
            "overall_cer": 14.2,
            "overall_bleu": 67.8,
            "per_dialect_wer": {"BE": 30.0, "ZH": 35.0, "VS": 32.5},
            "per_dialect_cer": {"BE": 12.0, "ZH": 16.0, "VS": 14.5},
            "per_dialect_bleu": {"BE": 70.0, "ZH": 65.0, "VS": 68.0}
        }
        fake_cache.get.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={"model_type": "whisper", "model": "base"}
        )

        assert response.status_code == 200
        data = response.json()

        # Check all required fields
        assert "model" in data
        assert "total_samples" in data
        assert "failed_samples" in data
        assert "overall_wer" in data
        assert "overall_cer" in data
        assert "overall_bleu" in data
        assert "per_dialect_wer" in data
        assert "per_dialect_cer" in data
        assert "per_dialect_bleu" in data

        # Check per-dialect structure
        assert len(data["per_dialect_wer"]) == 3
        assert "BE" in data["per_dialect_wer"]


class TestModelsEndpoint:
    """Tests for /models endpoint."""

    @pytest.mark.integration
    def test_list_models(self, client):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        # Expect at least one registry entry
        assert len(data) > 0
        first_entry = next(iter(data.values()))
        assert "type" in first_entry


@pytest.fixture
def temp_results(monkeypatch, temp_dir):
    """Create temporary results directory with sample model results."""
    from src.backend import endpoints
    
    # Create metrics directory structure
    metrics_root = temp_dir / "results" / "metrics"
    ts_dir = metrics_root / "20250101_120000"
    ts_dir.mkdir(parents=True)

    # Minimal model results payload
    payload = {
        "model_name": "whisper-large-v2",
        "timestamp": "20250101_120000",
        "results": {
            "overall_wer": 10.0,
            "overall_cer": 5.0,
            "overall_bleu": 80.0,
            "per_dialect_wer": {"BE": 12.0},
            "per_dialect_cer": {"BE": 6.0},
            "per_dialect_bleu": {"BE": 78.0},
            "samples": [
                {"audio_file": "a.wav", "dialect": "BE", "wer": 0.1, "cer": 0.05, "bleu": 90.0}
            ],
        }
    }

    result_file = ts_dir / "whisper-large-v2_results.json"
    result_file.write_text(json.dumps(payload), encoding="utf-8")

    # Patch RESULTS_DIR used by endpoints
    monkeypatch.setattr(endpoints, "RESULTS_DIR", temp_dir / "results")
    return result_file


class TestResultsEndpoints:
    """Tests for results listing and retrieval endpoints."""

    @pytest.mark.integration
    def test_list_results(self, client, temp_results):
        response = client.get("/api/results")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["model"] == "whisper-large-v2"
        assert data[0]["timestamp"] == "20250101_120000"

    @pytest.mark.integration
    def test_get_model_results(self, client, temp_results):
        response = client.get("/api/results/whisper-large-v2")
        assert response.status_code == 200
        data = response.json()
        assert data["results"]["overall_wer"] == 10.0

    @pytest.mark.integration
    def test_get_dialect_results(self, client, temp_results):
        response = client.get("/api/results/whisper-large-v2/BE")
        assert response.status_code == 200
        data = response.json()
        assert data["overall_wer"] == 12.0
        assert len(data["samples"]) == 1
        assert data["samples"][0]["dialect"] == "BE"


class TestModelCache:
    """Tests for model cache endpoints and behavior."""

    @pytest.mark.integration
    def test_cache_info(self, client, monkeypatch):
        from src.backend import endpoints
        from src.backend.model_cache import ModelCache

        cache = ModelCache(max_models=2)
        monkeypatch.setattr(endpoints, "get_model_cache", lambda: cache)

        response = client.get("/api/cache/info")
        assert response.status_code == 200
        data = response.json()
        assert data["max_models"] == 2
        assert data["size"] == 0

    @pytest.mark.integration
    def test_cache_clear(self, client, monkeypatch):
        from src.backend import endpoints
        from src.backend.model_cache import ModelCache

        cache = ModelCache(max_models=1)
        # Seed cache with dummy entry
        dummy = object()
        cache._cache[("whisper", "tiny", None)] = dummy  # type: ignore
        monkeypatch.setattr(endpoints, "get_model_cache", lambda: cache)

        response = client.post("/api/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert cache.info()["size"] == 0

    @pytest.mark.integration
    def test_model_caching_behavior(self, client, monkeypatch):
        from src.backend import endpoints

        class FakeEvaluator:
            def __init__(self):
                self.evaluate_calls = 0

            def evaluate_dataset(self, **kwargs):
                self.evaluate_calls += 1
                return {
                    "total_samples": 1,
                    "failed_samples": 0,
                    "overall_wer": 1.0,
                    "overall_cer": 1.0,
                    "overall_bleu": 1.0,
                    "per_dialect_wer": {},
                    "per_dialect_cer": {},
                    "per_dialect_bleu": {},
                }

        class FakeCache:
            def __init__(self):
                self.evaluator = None
                self.load_count = 0

            def get(self, model_type: str, model_name: str, lm_path=None):
                if self.evaluator is None:
                    self.evaluator = FakeEvaluator()
                    self.load_count += 1
                return self.evaluator

            def info(self):
                return {"size": 1, "max_models": 1, "entries": []}

            def clear(self):
                self.evaluator = None

        fake_cache = FakeCache()
        monkeypatch.setattr(endpoints, "get_model_cache", lambda: fake_cache)

        payload = {"model_type": "whisper", "model": "tiny"}
        r1 = client.post("/api/evaluate", json=payload)
        r2 = client.post("/api/evaluate", json=payload)

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert fake_cache.load_count == 1
        assert fake_cache.evaluator.evaluate_calls == 2

class TestCacheInfoEndpoint:
    """Comprehensive tests for /cache/info endpoint."""

    @pytest.mark.integration
    def test_cache_info_empty_cache(self, client, monkeypatch):
        """Test cache info with empty cache."""
        from src.backend import endpoints
        from src.backend.model_cache import ModelCache

        cache = ModelCache(max_models=5)
        monkeypatch.setattr(endpoints, "get_model_cache", lambda: cache)

        response = client.get("/api/cache/info")
        assert response.status_code == 200
        data = response.json()
        assert data["max_models"] == 5
        assert data["size"] == 0
        assert data["entries"] == []

    @pytest.mark.integration
    def test_cache_info_with_entries(self, client, monkeypatch):
        """Test cache info with cached models."""
        from src.backend import endpoints
        from src.backend.model_cache import ModelCache

        cache = ModelCache(max_models=3)
        # Manually add cache entries
        cache._cache[("whisper", "tiny", None)] = Mock()
        cache._cache[("wav2vec2", "xlsr-53", "/path/to/lm.arpa")] = Mock()
        monkeypatch.setattr(endpoints, "get_model_cache", lambda: cache)

        response = client.get("/api/cache/info")
        assert response.status_code == 200
        data = response.json()
        assert data["max_models"] == 3
        assert data["size"] == 2
        assert len(data["entries"]) == 2

    @pytest.mark.integration
    def test_cache_info_entry_structure(self, client, monkeypatch):
        """Test that cache info entries have correct structure."""
        from src.backend import endpoints
        from src.backend.model_cache import ModelCache

        cache = ModelCache()
        cache._cache[("mms", "1b-all", None)] = Mock()
        monkeypatch.setattr(endpoints, "get_model_cache", lambda: cache)

        response = client.get("/api/cache/info")
        data = response.json()
        entry = data["entries"][0]
        assert "model_type" in entry
        assert "model_name" in entry
        assert "lm_path" in entry
        assert entry["model_type"] == "mms"
        assert entry["model_name"] == "1b-all"
        assert entry["lm_path"] is None


class TestCacheClearEndpoint:
    """Comprehensive tests for /cache/clear endpoint."""

    @pytest.mark.integration
    def test_cache_clear_success(self, client, monkeypatch):
        """Test successful cache clear."""
        from src.backend import endpoints
        from src.backend.model_cache import ModelCache

        cache = ModelCache()
        cache._cache[("whisper", "tiny", None)] = Mock()
        assert len(cache._cache) == 1
        monkeypatch.setattr(endpoints, "get_model_cache", lambda: cache)

        response = client.post("/api/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data
        assert len(cache._cache) == 0

    @pytest.mark.integration
    def test_cache_clear_already_empty(self, client, monkeypatch):
        """Test clearing an already empty cache."""
        from src.backend import endpoints
        from src.backend.model_cache import ModelCache

        cache = ModelCache()
        assert len(cache._cache) == 0
        monkeypatch.setattr(endpoints, "get_model_cache", lambda: cache)

        response = client.post("/api/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.integration
    def test_cache_clear_multiple_entries(self, client, monkeypatch):
        """Test clearing cache with multiple entries."""
        from src.backend import endpoints
        from src.backend.model_cache import ModelCache

        cache = ModelCache(max_models=10)
        for i in range(5):
            cache._cache[(f"model_type_{i}", f"model_{i}", None)] = Mock()
        assert len(cache._cache) == 5
        monkeypatch.setattr(endpoints, "get_model_cache", lambda: cache)

        response = client.post("/api/cache/clear")
        assert response.status_code == 200
        assert len(cache._cache) == 0


class TestResultsEndpointEdgeCases:
    """Edge case tests for results endpoints."""

    @pytest.mark.integration
    def test_get_results_no_metrics_dir(self, client, monkeypatch, tmp_path):
        """Test /results when metrics directory doesn't exist."""
        from src.backend import endpoints

        # Point to non-existent directory
        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "nonexistent")

        response = client.get("/api/results")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.integration
    def test_get_results_empty_metrics_dir(self, client, monkeypatch, tmp_path):
        """Test /results with empty metrics directory."""
        from src.backend import endpoints

        metrics_dir = tmp_path / "results" / "metrics"
        metrics_dir.mkdir(parents=True)
        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.integration
    def test_get_model_results_with_timestamp(self, client, monkeypatch, tmp_path):
        """Test /results/{model} with explicit timestamp."""
        from src.backend import endpoints

        results_dir = tmp_path / "results" / "metrics" / "20250101_120000"
        results_dir.mkdir(parents=True)

        payload = {
            "model_name": "test-model",
            "results": {
                "overall_wer": 25.0,
                "overall_cer": 10.0,
                "overall_bleu": 75.0
            }
        }
        result_file = results_dir / "test-model_results.json"
        result_file.write_text(json.dumps(payload))

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/test-model?timestamp=20250101_120000")
        assert response.status_code == 200
        data = response.json()
        assert data["results"]["overall_wer"] == 25.0

    @pytest.mark.integration
    def test_get_model_results_timestamp_not_found(self, client, monkeypatch, tmp_path):
        """Test /results/{model} with invalid timestamp."""
        from src.backend import endpoints

        results_dir = tmp_path / "results" / "metrics"
        results_dir.mkdir(parents=True)
        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/nonexistent-model?timestamp=20250101_120000")
        assert response.status_code == 404

    @pytest.mark.integration
    def test_get_model_results_no_timestamp_directory_missing(self, client, monkeypatch, tmp_path):
        """Test /results/{model} without timestamp when no results dir."""
        from src.backend import endpoints

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/test-model")
        assert response.status_code == 404
        data = response.json()
        assert "No results directory found" in data["detail"]

    @pytest.mark.integration
    def test_get_model_results_finds_most_recent(self, client, monkeypatch, tmp_path):
        """Test /results/{model} finds most recent timestamp when not specified."""
        from src.backend import endpoints

        # Create multiple timestamps
        ts_old = tmp_path / "results" / "metrics" / "20250101_120000"
        ts_new = tmp_path / "results" / "metrics" / "20250102_120000"
        ts_old.mkdir(parents=True)
        ts_new.mkdir(parents=True)

        old_data = {"results": {"overall_wer": 30.0}}
        new_data = {"results": {"overall_wer": 20.0}}

        (ts_old / "model_results.json").write_text(json.dumps(old_data))
        (ts_new / "model_results.json").write_text(json.dumps(new_data))

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/model")
        assert response.status_code == 200
        data = response.json()
        # Should get the most recent (newest timestamp)
        assert data["results"]["overall_wer"] == 20.0

    @pytest.mark.integration
    def test_get_dialect_results_with_timestamp(self, client, monkeypatch, tmp_path):
        """Test /results/{model}/{dialect} with explicit timestamp."""
        from src.backend import endpoints

        results_dir = tmp_path / "results" / "metrics" / "20250101_120000"
        results_dir.mkdir(parents=True)

        payload = {
            "results": {
                "per_dialect_wer": {"BE": 22.0, "ZH": 25.0},
                "per_dialect_cer": {"BE": 10.0, "ZH": 12.0},
                "per_dialect_bleu": {"BE": 78.0, "ZH": 76.0},
                "samples": [
                    {"dialect": "BE", "wer": 0.2, "cer": 0.1, "bleu": 80.0},
                    {"dialect": "ZH", "wer": 0.25, "cer": 0.12, "bleu": 76.0}
                ]
            }
        }
        result_file = results_dir / "test-model_results.json"
        result_file.write_text(json.dumps(payload))

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/test-model/BE?timestamp=20250101_120000")
        assert response.status_code == 200
        data = response.json()
        assert data["overall_wer"] == 22.0
        assert len(data["samples"]) == 1
        assert data["samples"][0]["dialect"] == "BE"

    @pytest.mark.integration
    def test_get_dialect_results_dialect_not_found(self, client, monkeypatch, tmp_path):
        """Test /results/{model}/{dialect} with invalid dialect."""
        from src.backend import endpoints

        results_dir = tmp_path / "results" / "metrics" / "20250101_120000"
        results_dir.mkdir(parents=True)

        payload = {
            "results": {
                "per_dialect_wer": {"BE": 22.0},
                "per_dialect_cer": {"BE": 10.0},
                "per_dialect_bleu": {"BE": 78.0},
                "samples": []
            }
        }
        result_file = results_dir / "test-model_results.json"
        result_file.write_text(json.dumps(payload))

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/test-model/INVALID?timestamp=20250101_120000")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.integration
    def test_get_dialect_results_no_samples_for_dialect(self, client, monkeypatch, tmp_path):
        """Test /results/{model}/{dialect} with no samples for dialect."""
        from src.backend import endpoints

        results_dir = tmp_path / "results" / "metrics" / "20250101_120000"
        results_dir.mkdir(parents=True)

        payload = {
            "results": {
                "per_dialect_wer": {"BE": 22.0},
                "per_dialect_cer": {"BE": 10.0},
                "per_dialect_bleu": {"BE": 78.0},
                "samples": [
                    {"dialect": "ZH", "wer": 0.25, "cer": 0.12, "bleu": 76.0}
                ]
            }
        }
        result_file = results_dir / "test-model_results.json"
        result_file.write_text(json.dumps(payload))

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/test-model/BE?timestamp=20250101_120000")
        assert response.status_code == 200
        data = response.json()
        assert data["overall_wer"] == 22.0
        assert data["samples"] == []

    @pytest.mark.integration
    def test_get_dialect_results_json_read_error(self, client, monkeypatch, tmp_path):
        """Test /results/{model}/{dialect} with malformed JSON."""
        from src.backend import endpoints

        results_dir = tmp_path / "results" / "metrics" / "20250101_120000"
        results_dir.mkdir(parents=True)

        result_file = results_dir / "test-model_results.json"
        result_file.write_text("invalid json {")

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/test-model/BE?timestamp=20250101_120000")
        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"].lower()

    @pytest.mark.integration
    def test_get_model_results_json_read_error(self, client, monkeypatch, tmp_path):
        """Test /results/{model} with malformed JSON."""
        from src.backend import endpoints

        results_dir = tmp_path / "results" / "metrics" / "20250101_120000"
        results_dir.mkdir(parents=True)

        result_file = results_dir / "test-model_results.json"
        result_file.write_text("invalid json {")

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/test-model?timestamp=20250101_120000")
        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"].lower()

    @pytest.mark.integration
    def test_get_results_mixed_files_and_dirs(self, client, monkeypatch, tmp_path):
        """Test /results when metrics dir has mixed files and directories."""
        from src.backend import endpoints

        metrics_dir = tmp_path / "results" / "metrics"
        ts_dir = metrics_dir / "20250101_120000"
        ts_dir.mkdir(parents=True)

        # Create a file (not directory) in metrics dir
        (metrics_dir / "some_file.txt").write_text("ignored")

        payload = {"results": {"overall_wer": 25.0}}
        (ts_dir / "model_results.json").write_text(json.dumps(payload))

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results")
        assert response.status_code == 200
        data = response.json()
        # Should only find the one result, ignoring the file
        assert len(data) == 1


class TestModelsEndpointEdgeCases:
    """Edge case tests for /models endpoint."""

    @pytest.mark.integration
    def test_models_endpoint_returns_dict(self, client):
        """Test /models endpoint returns valid model dictionary."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)



class TestResultsEndpointFileHandling:
    """Tests for file handling edge cases in results endpoints."""

    @pytest.mark.integration
    def test_get_model_results_file_encoding(self, client, monkeypatch, tmp_path):
        """Test /results/{model} handles various file encodings."""
        from src.backend import endpoints

        results_dir = tmp_path / "results" / "metrics" / "20250101_120000"
        results_dir.mkdir(parents=True)

        payload = {
            "model_name": "test-model",
            "results": {
                "overall_wer": 25.0,
                "overall_cer": 10.0,
                "overall_bleu": 75.0,
                "per_dialect_wer": {},
                "per_dialect_cer": {},
                "per_dialect_bleu": {},
                "samples": []
            }
        }
        result_file = results_dir / "test-model_results.json"
        result_file.write_text(json.dumps(payload), encoding="utf-8")

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/test-model?timestamp=20250101_120000")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_get_dialect_results_missing_per_dialect_fields(self, client, monkeypatch, tmp_path):
        """Test /results/{model}/{dialect} with missing per-dialect fields."""
        from src.backend import endpoints

        results_dir = tmp_path / "results" / "metrics" / "20250101_120000"
        results_dir.mkdir(parents=True)

        # Minimal payload without per_dialect fields
        payload = {
            "results": {
                "per_dialect_wer": {},
                "per_dialect_cer": {},
                "per_dialect_bleu": {},
                "samples": []
            }
        }
        result_file = results_dir / "test-model_results.json"
        result_file.write_text(json.dumps(payload))

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/test-model/BE?timestamp=20250101_120000")
        assert response.status_code == 404

    @pytest.mark.integration
    def test_get_dialect_results_with_complex_samples(self, client, monkeypatch, tmp_path):
        """Test /results/{model}/{dialect} with complex sample data."""
        from src.backend import endpoints

        results_dir = tmp_path / "results" / "metrics" / "20250101_120000"
        results_dir.mkdir(parents=True)

        payload = {
            "results": {
                "per_dialect_wer": {"BE": 22.0},
                "per_dialect_cer": {"BE": 10.0},
                "per_dialect_bleu": {"BE": 78.0},
                "samples": [
                    {
                        "audio_file": "file1.wav",
                        "dialect": "BE",
                        "wer": 0.2,
                        "cer": 0.1,
                        "bleu": 80.0,
                        "reference": "hello world",
                        "hypothesis": "hallo world"
                    },
                    {
                        "audio_file": "file2.wav",
                        "dialect": "ZH",
                        "wer": 0.25,
                        "cer": 0.12,
                        "bleu": 76.0,
                        "reference": "good morning",
                        "hypothesis": "good morning"
                    }
                ]
            }
        }
        result_file = results_dir / "test-model_results.json"
        result_file.write_text(json.dumps(payload))

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results/test-model/BE?timestamp=20250101_120000")
        assert response.status_code == 200
        data = response.json()
        assert len(data["samples"]) == 1
        assert data["samples"][0]["audio_file"] == "file1.wav"


class TestResultsEndpointTimestampHandling:
    """Tests for timestamp handling in results endpoints."""

    @pytest.mark.integration
    def test_results_with_numeric_timestamp_sorting(self, client, monkeypatch, tmp_path):
        """Test that timestamps are sorted numerically."""
        from src.backend import endpoints

        # Create timestamps in non-chronological order
        timestamps = ["20250101_120000", "20250110_090000", "20250103_180000"]
        metrics_root = tmp_path / "results" / "metrics"

        for ts in timestamps:
            ts_dir = metrics_root / ts
            ts_dir.mkdir(parents=True)
            payload = {"results": {"overall_wer": 25.0}}
            (ts_dir / "model_results.json").write_text(json.dumps(payload))

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        response = client.get("/api/results")
        assert response.status_code == 200
        data = response.json()

        # Should be sorted descending (newest first)
        assert len(data) == 3
        assert data[0]["timestamp"] == "20250110_090000"
        assert data[1]["timestamp"] == "20250103_180000"
        assert data[2]["timestamp"] == "20250101_120000"

    @pytest.mark.integration
    def test_results_prefers_exact_timestamp(self, client, monkeypatch, tmp_path):
        """Test that exact timestamp is used when provided."""
        from src.backend import endpoints

        ts1_dir = tmp_path / "results" / "metrics" / "20250101_120000"
        ts2_dir = tmp_path / "results" / "metrics" / "20250102_120000"
        ts1_dir.mkdir(parents=True)
        ts2_dir.mkdir(parents=True)

        payload1 = {"results": {"overall_wer": 30.0}}
        payload2 = {"results": {"overall_wer": 20.0}}

        (ts1_dir / "model_results.json").write_text(json.dumps(payload1))
        (ts2_dir / "model_results.json").write_text(json.dumps(payload2))

        monkeypatch.setattr(endpoints, "RESULTS_DIR", tmp_path / "results")

        # Request with specific timestamp
        response = client.get("/api/results/model?timestamp=20250101_120000")
        assert response.status_code == 200
        data = response.json()
        assert data["results"]["overall_wer"] == 30.0


class TestEvaluateEndpointEdgeCases:
    """Edge case tests for /evaluate endpoint."""

    @pytest.mark.integration
    def test_evaluate_dataset_with_zero_samples(self, client, fake_cache):
        """Test evaluation with zero samples."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 0,
            "failed_samples": 0,
            "overall_wer": 0.0,
            "overall_cer": 0.0,
            "overall_bleu": 0.0,
            "per_dialect_wer": {},
            "per_dialect_cer": {},
            "per_dialect_bleu": {}
        }
        fake_cache.get.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={"model_type": "whisper", "model": "tiny"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_samples"] == 0

    @pytest.mark.integration
    def test_evaluate_all_samples_failed(self, client, fake_cache):
        """Test evaluation where all samples fail."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 10,
            "failed_samples": 10,
            "overall_wer": 100.0,
            "overall_cer": 100.0,
            "overall_bleu": 0.0,
            "per_dialect_wer": {},
            "per_dialect_cer": {},
            "per_dialect_bleu": {}
        }
        fake_cache.get.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={"model_type": "whisper", "model": "base"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["failed_samples"] == 10
        assert data["total_samples"] == 10

    @pytest.mark.integration
    def test_evaluate_extreme_metric_values(self, client, fake_cache):
        """Test evaluation with extreme metric values."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 1,
            "failed_samples": 0,
            "overall_wer": 0.0,
            "overall_cer": 0.0,
            "overall_bleu": 100.0,
            "per_dialect_wer": {"BE": 0.0},
            "per_dialect_cer": {"BE": 0.0},
            "per_dialect_bleu": {"BE": 100.0}
        }
        fake_cache.get.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={"model_type": "whisper", "model": "large"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["overall_wer"] == 0.0
        assert data["overall_bleu"] == 100.0

    @pytest.mark.integration
    def test_evaluate_many_dialects(self, client, fake_cache):
        """Test evaluation with many dialect results."""
        mock_evaluator = Mock()
        dialects = ["BE", "ZH", "VS", "GE", "FR", "IT"]
        dialect_wer = {d: 20.0 + i for i, d in enumerate(dialects)}
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 100,
            "failed_samples": 0,
            "overall_wer": 22.5,
            "overall_cer": 10.0,
            "overall_bleu": 70.0,
            "per_dialect_wer": dialect_wer,
            "per_dialect_cer": {d: 10.0 for d in dialects},
            "per_dialect_bleu": {d: 70.0 for d in dialects}
        }
        fake_cache.get.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={"model_type": "wav2vec2", "model": "xlsr"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["per_dialect_wer"]) == 6
        assert "BE" in data["per_dialect_wer"]