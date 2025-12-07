"""End-to-end tests for API workflow."""
import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient


class TestAPIWorkflow:
    """Test API workflow end-to-end."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        from src.backend.endpoints import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router, prefix="/api")

        return TestClient(app)

    @pytest.mark.e2e
    @patch('src.backend.endpoints.ASREvaluator')
    def test_complete_api_evaluation_flow(self, mock_evaluator_class, client):
        """Test complete API evaluation flow."""
        # Setup mock
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 100,
            "failed_samples": 5,
            "overall_wer": 28.5,
            "overall_cer": 12.3,
            "overall_bleu": 68.7,
            "per_dialect_wer": {"BE": 25.0, "ZH": 30.0, "VS": 32.0},
            "per_dialect_cer": {"BE": 10.0, "ZH": 13.0, "VS": 15.0},
            "per_dialect_bleu": {"BE": 72.0, "ZH": 67.0, "VS": 65.0}
        }
        mock_evaluator_class.return_value = mock_evaluator

        # Make request
        response = client.post(
            "/api/evaluate",
            json={
                "model_type": "whisper",
                "model": "medium",
                "limit": 100
            }
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Check all fields present
        assert data["model"] == "medium"
        assert data["total_samples"] == 100
        assert data["failed_samples"] == 5
        assert data["overall_wer"] == 28.5
        assert data["overall_cer"] == 12.3
        assert data["overall_bleu"] == 68.7

        # Check per-dialect metrics
        assert len(data["per_dialect_wer"]) == 3
        assert data["per_dialect_wer"]["BE"] == 25.0

    @pytest.mark.e2e
    @patch('src.backend.endpoints.ASREvaluator')
    def test_api_multiple_model_types(self, mock_evaluator_class, client):
        """Test API with different model types."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 50,
            "failed_samples": 0,
            "overall_wer": 30.0,
            "overall_cer": 15.0,
            "overall_bleu": 65.0,
            "per_dialect_wer": {"BE": 30.0},
            "per_dialect_cer": {"BE": 15.0},
            "per_dialect_bleu": {"BE": 65.0}
        }
        mock_evaluator_class.return_value = mock_evaluator

        model_configs = [
            {"model_type": "whisper", "model": "tiny"},
            {"model_type": "wav2vec2", "model": "facebook/wav2vec2-base"},
            {"model_type": "mms", "model": "facebook/mms-1b-all"}
        ]

        for config in model_configs:
            response = client.post("/api/evaluate", json=config)

            assert response.status_code == 200, f"Failed for {config['model_type']}"
            data = response.json()
            assert data["model"] == config["model"]

    @pytest.mark.e2e
    def test_api_input_validation(self, client):
        """Test API input validation scenarios."""
        # Missing required fields
        response = client.post("/api/evaluate", json={})
        assert response.status_code == 422

        # Invalid model type
        response = client.post(
            "/api/evaluate",
            json={"model_type": "invalid", "model": "test"}
        )
        assert response.status_code == 422

        # Negative limit
        response = client.post(
            "/api/evaluate",
            json={"model_type": "whisper", "model": "tiny", "limit": -1}
        )
        assert response.status_code == 422

        # Zero limit
        response = client.post(
            "/api/evaluate",
            json={"model_type": "whisper", "model": "tiny", "limit": 0}
        )
        assert response.status_code == 422

    @pytest.mark.e2e
    @patch('src.backend.endpoints.ASREvaluator')
    def test_api_error_handling(self, mock_evaluator_class, client):
        """Test API error handling."""
        mock_evaluator = Mock()
        mock_evaluator.load_model.side_effect = RuntimeError("Model not found")
        mock_evaluator_class.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={"model_type": "whisper", "model": "nonexistent"}
        )

        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()


class TestAPIResponseValidation:
    """Test API response validation and format."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        from src.backend.endpoints import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router, prefix="/api")

        return TestClient(app)

    @pytest.mark.e2e
    @patch('src.backend.endpoints.ASREvaluator')
    def test_response_schema_validation(self, mock_evaluator_class, client):
        """Test that response matches expected schema."""
        from src.backend.models import EvaluateResponse

        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 25,
            "failed_samples": 2,
            "overall_wer": 32.5,
            "overall_cer": 14.2,
            "overall_bleu": 64.8,
            "per_dialect_wer": {"BE": 30.0, "ZH": 35.0},
            "per_dialect_cer": {"BE": 12.0, "ZH": 16.0},
            "per_dialect_bleu": {"BE": 68.0, "ZH": 62.0}
        }
        mock_evaluator_class.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={"model_type": "whisper", "model": "base"}
        )

        assert response.status_code == 200

        # Validate response can be parsed as EvaluateResponse
        data = response.json()
        response_obj = EvaluateResponse(**data)

        assert response_obj.model == "base"
        assert response_obj.total_samples == 25
        assert response_obj.failed_samples == 2

    @pytest.mark.e2e
    @patch('src.backend.endpoints.ASREvaluator')
    def test_response_metric_ranges(self, mock_evaluator_class, client):
        """Test that metrics are in expected ranges."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_dataset.return_value = {
            "total_samples": 10,
            "failed_samples": 0,
            "overall_wer": 0.0,  # Perfect
            "overall_cer": 0.0,
            "overall_bleu": 100.0,
            "per_dialect_wer": {"BE": 0.0},
            "per_dialect_cer": {"BE": 0.0},
            "per_dialect_bleu": {"BE": 100.0}
        }
        mock_evaluator_class.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={"model_type": "whisper", "model": "tiny"}
        )

        data = response.json()

        # WER and CER should be >= 0
        assert data["overall_wer"] >= 0
        assert data["overall_cer"] >= 0

        # BLEU should be between 0 and 100
        assert 0 <= data["overall_bleu"] <= 100

        # Samples counts should be non-negative
        assert data["total_samples"] >= 0
        assert data["failed_samples"] >= 0
        assert data["failed_samples"] <= data["total_samples"]
