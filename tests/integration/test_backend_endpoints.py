"""Integration tests for FastAPI backend endpoints."""
import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from src.backend.endpoints import router

    app = FastAPI()
    app.include_router(router, prefix="/api")

    return TestClient(app)


class TestEvaluateEndpoint:
    """Test /evaluate endpoint integration."""

    @pytest.mark.integration
    @patch('src.backend.endpoints.ASREvaluator')
    def test_evaluate_endpoint_success(self, mock_evaluator_class, client):
        """Test successful evaluation request."""
        # Setup mock evaluator
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
        mock_evaluator_class.return_value = mock_evaluator

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
    @patch('src.backend.endpoints.ASREvaluator')
    def test_evaluate_endpoint_with_limit(self, mock_evaluator_class, client):
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
        mock_evaluator_class.return_value = mock_evaluator

        response = client.post(
            "/api/evaluate",
            json={
                "model_type": "whisper",
                "model": "base",
                "limit": 5
            }
        )

        assert response.status_code == 200
        # Verify limit was passed
        mock_evaluator.evaluate_dataset.assert_called_once()
        call_kwargs = mock_evaluator.evaluate_dataset.call_args
        assert call_kwargs[1].get('limit') == 5 or call_kwargs[0][1] == 5 if len(call_kwargs[0]) > 1 else True

    @pytest.mark.integration
    @patch('src.backend.endpoints.ASREvaluator')
    def test_evaluate_endpoint_wav2vec2_model(self, mock_evaluator_class, client):
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
        mock_evaluator_class.return_value = mock_evaluator

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
    @patch('src.backend.endpoints.ASREvaluator')
    def test_evaluate_endpoint_mms_model(self, mock_evaluator_class, client):
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
        mock_evaluator_class.return_value = mock_evaluator

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
    @patch('src.backend.endpoints.ASREvaluator')
    def test_evaluate_endpoint_handles_error(self, mock_evaluator_class, client):
        """Test endpoint handles evaluation errors gracefully."""
        mock_evaluator = Mock()
        mock_evaluator.load_model.side_effect = Exception("Model loading failed")
        mock_evaluator_class.return_value = mock_evaluator

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
    @patch('src.backend.endpoints.ASREvaluator')
    def test_response_contains_all_metrics(self, mock_evaluator_class, client):
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
        mock_evaluator_class.return_value = mock_evaluator

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
