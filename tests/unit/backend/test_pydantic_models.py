"""Unit tests for Pydantic models."""
import pytest
from pydantic import ValidationError
from src.backend.models import EvaluateRequest, EvaluateResponse


class TestEvaluateRequest:
    """Test suite for EvaluateRequest model."""

    @pytest.mark.unit
    def test_valid_request_minimal(self):
        """Test valid request with minimal required fields."""
        request = EvaluateRequest(model="tiny")

        assert request.model == "tiny"
        assert request.model_type == "whisper"  # Default
        assert request.limit is None

    @pytest.mark.unit
    def test_valid_request_with_all_fields(self):
        """Test valid request with all fields."""
        request = EvaluateRequest(
            model_type="wav2vec2",
            model="facebook/wav2vec2-large-xlsr-53-german",
            limit=100
        )

        assert request.model_type == "wav2vec2"
        assert request.model == "facebook/wav2vec2-large-xlsr-53-german"
        assert request.limit == 100

    @pytest.mark.unit
    def test_model_type_default_whisper(self):
        """Test model_type defaults to whisper."""
        request = EvaluateRequest(model="base")

        assert request.model_type == "whisper"

    @pytest.mark.unit
    @pytest.mark.parametrize("model_type", ["whisper", "wav2vec2", "mms"])
    def test_valid_model_types(self, model_type):
        """Test all valid model types are accepted."""
        request = EvaluateRequest(model_type=model_type, model="test")

        assert request.model_type == model_type

    @pytest.mark.unit
    def test_invalid_model_type_raises_error(self):
        """Test invalid model_type raises ValidationError."""
        with pytest.raises(ValidationError):
            EvaluateRequest(model_type="invalid", model="test")

    @pytest.mark.unit
    def test_limit_optional(self):
        """Test limit is optional."""
        request = EvaluateRequest(model="medium")

        assert request.limit is None

    @pytest.mark.unit
    def test_limit_positive_integer(self):
        """Test limit accepts positive integers."""
        request = EvaluateRequest(model="test", limit=50)

        assert request.limit == 50

    @pytest.mark.unit
    def test_limit_zero_raises_error(self):
        """Test limit=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            EvaluateRequest(model="test", limit=0)

    @pytest.mark.unit
    def test_limit_negative_raises_error(self):
        """Test negative limit raises ValidationError."""
        with pytest.raises(ValidationError):
            EvaluateRequest(model="test", limit=-10)

    @pytest.mark.unit
    def test_missing_model_raises_error(self):
        """Test missing model field raises ValidationError."""
        with pytest.raises(ValidationError):
            EvaluateRequest(model_type="whisper")

    @pytest.mark.unit
    @pytest.mark.parametrize("model_name", [
        "tiny", "base", "small", "medium", "large", "large-v2"
    ])
    def test_whisper_model_names(self, model_name):
        """Test various Whisper model names are accepted."""
        request = EvaluateRequest(model_type="whisper", model=model_name)

        assert request.model == model_name

    @pytest.mark.unit
    def test_wav2vec2_model_names(self):
        """Test Wav2Vec2 model names are accepted."""
        request = EvaluateRequest(
            model_type="wav2vec2",
            model="facebook/wav2vec2-large-xlsr-53-german"
        )

        assert "wav2vec2" in request.model

    @pytest.mark.unit
    def test_mms_model_names(self):
        """Test MMS model names are accepted."""
        request = EvaluateRequest(
            model_type="mms",
            model="facebook/mms-1b-all"
        )

        assert "mms" in request.model


class TestEvaluateResponse:
    """Test suite for EvaluateResponse model."""

    @pytest.fixture
    def valid_response_data(self):
        """Valid response data for testing."""
        return {
            "model": "whisper-medium",
            "total_samples": 100,
            "failed_samples": 2,
            "overall_wer": 35.5,
            "overall_cer": 15.2,
            "overall_bleu": 65.3,
            "per_dialect_wer": {"BE": 30.0, "ZH": 35.0},
            "per_dialect_cer": {"BE": 12.0, "ZH": 18.0},
            "per_dialect_bleu": {"BE": 70.0, "ZH": 60.0}
        }

    @pytest.mark.unit
    def test_valid_response(self, valid_response_data):
        """Test valid response creation."""
        response = EvaluateResponse(**valid_response_data)

        assert response.model == "whisper-medium"
        assert response.total_samples == 100
        assert response.failed_samples == 2
        assert response.overall_wer == 35.5
        assert response.overall_cer == 15.2
        assert response.overall_bleu == 65.3

    @pytest.mark.unit
    def test_per_dialect_metrics(self, valid_response_data):
        """Test per-dialect metrics are stored correctly."""
        response = EvaluateResponse(**valid_response_data)

        assert len(response.per_dialect_wer) == 2
        assert response.per_dialect_wer["BE"] == 30.0
        assert response.per_dialect_wer["ZH"] == 35.0

    @pytest.mark.unit
    def test_empty_per_dialect_metrics(self):
        """Test response with empty per-dialect metrics."""
        response = EvaluateResponse(
            model="test",
            total_samples=10,
            failed_samples=0,
            overall_wer=25.0,
            overall_cer=10.0,
            overall_bleu=75.0,
            per_dialect_wer={},
            per_dialect_cer={},
            per_dialect_bleu={}
        )

        assert response.per_dialect_wer == {}
        assert response.per_dialect_cer == {}
        assert response.per_dialect_bleu == {}

    @pytest.mark.unit
    def test_zero_failed_samples(self, valid_response_data):
        """Test response with zero failed samples."""
        valid_response_data["failed_samples"] = 0
        response = EvaluateResponse(**valid_response_data)

        assert response.failed_samples == 0

    @pytest.mark.unit
    def test_all_samples_failed(self):
        """Test response where all samples failed."""
        response = EvaluateResponse(
            model="test",
            total_samples=10,
            failed_samples=10,
            overall_wer=0.0,
            overall_cer=0.0,
            overall_bleu=0.0,
            per_dialect_wer={},
            per_dialect_cer={},
            per_dialect_bleu={}
        )

        assert response.total_samples == 10
        assert response.failed_samples == 10

    @pytest.mark.unit
    def test_perfect_scores(self):
        """Test response with perfect scores (WER/CER=0, BLEU=100)."""
        response = EvaluateResponse(
            model="perfect-model",
            total_samples=50,
            failed_samples=0,
            overall_wer=0.0,
            overall_cer=0.0,
            overall_bleu=100.0,
            per_dialect_wer={"BE": 0.0},
            per_dialect_cer={"BE": 0.0},
            per_dialect_bleu={"BE": 100.0}
        )

        assert response.overall_wer == 0.0
        assert response.overall_bleu == 100.0

    @pytest.mark.unit
    def test_many_dialects(self):
        """Test response with many dialects."""
        dialects = {f"dialect_{i}": float(i * 5) for i in range(10)}

        response = EvaluateResponse(
            model="test",
            total_samples=1000,
            failed_samples=50,
            overall_wer=30.0,
            overall_cer=12.0,
            overall_bleu=68.0,
            per_dialect_wer=dialects,
            per_dialect_cer=dialects,
            per_dialect_bleu=dialects
        )

        assert len(response.per_dialect_wer) == 10

    @pytest.mark.unit
    def test_response_model_conversion(self, valid_response_data):
        """Test response can be converted to dict."""
        response = EvaluateResponse(**valid_response_data)
        response_dict = response.model_dump()

        assert isinstance(response_dict, dict)
        assert "model" in response_dict
        assert "overall_wer" in response_dict

    @pytest.mark.unit
    def test_response_json_serialization(self, valid_response_data):
        """Test response can be serialized to JSON."""
        response = EvaluateResponse(**valid_response_data)
        json_str = response.model_dump_json()

        assert isinstance(json_str, str)
        assert "whisper-medium" in json_str

    @pytest.mark.unit
    def test_missing_required_fields_raises_error(self):
        """Test missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            EvaluateResponse(model="test")  # Missing all other fields

    @pytest.mark.unit
    @pytest.mark.parametrize("metric_value", [0.0, 50.0, 100.0, 150.0])
    def test_metric_value_ranges(self, metric_value):
        """Test various metric values are accepted (no validation on range)."""
        response = EvaluateResponse(
            model="test",
            total_samples=10,
            failed_samples=0,
            overall_wer=metric_value,
            overall_cer=metric_value,
            overall_bleu=metric_value,
            per_dialect_wer={"BE": metric_value},
            per_dialect_cer={"BE": metric_value},
            per_dialect_bleu={"BE": metric_value}
        )

        assert response.overall_wer == metric_value
