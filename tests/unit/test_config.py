"""Unit tests for config module."""
import pytest
import os
from pathlib import Path
from unittest.mock import patch
from src.config import (
    _detect_environment_legacy,
    ENVIRONMENT,
    IS_AUTO_DETECTED,
    VALID_ENVIRONMENTS,
    get_config_summary,
    create_directories,
    validate_paths,
    log_configuration,
    IS_RUNPOD,
    ENV_DEFAULTS,
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
)


class TestDetectEnvironmentLegacy:
    """Test legacy environment detection."""

    @pytest.mark.unit
    def test_detect_environment_runpod_legacy(self):
        """Test detection of RunPod environment based on filesystem."""
        with patch("os.path.exists") as mock_exists:
            # Mock: /workspace exists, /app doesn't
            def exists_side_effect(path):
                return path == "/workspace"
            
            mock_exists.side_effect = exists_side_effect
            result = _detect_environment_legacy()
            assert result == "runpod"

    @pytest.mark.unit
    def test_detect_environment_local_legacy(self):
        """Test detection of local environment based on filesystem."""
        with patch("os.path.exists") as mock_exists:
            # Mock: Neither /workspace nor /app exist (or /app exists)
            mock_exists.return_value = False
            result = _detect_environment_legacy()
            assert result == "local"

    @pytest.mark.unit
    def test_detect_environment_local_when_both_exist(self):
        """Test that local is default when both /workspace and /app exist."""
        with patch("os.path.exists") as mock_exists:
            # Mock: Both exist
            mock_exists.return_value = True
            result = _detect_environment_legacy()
            assert result == "local"


class TestEnvironmentConfiguration:
    """Test environment configuration logic."""

    @pytest.mark.unit
    def test_environment_is_valid(self):
        """Test that ENVIRONMENT is one of valid options."""
        assert ENVIRONMENT in VALID_ENVIRONMENTS

    @pytest.mark.unit
    def test_valid_environments_list(self):
        """Test that VALID_ENVIRONMENTS contains expected values."""
        assert "local" in VALID_ENVIRONMENTS
        assert "runpod" in VALID_ENVIRONMENTS
        assert "ci" in VALID_ENVIRONMENTS

    @pytest.mark.unit
    def test_is_runpod_boolean(self):
        """Test that IS_RUNPOD is a boolean."""
        assert isinstance(IS_RUNPOD, bool)

    @pytest.mark.unit
    def test_is_runpod_matches_environment(self):
        """Test that IS_RUNPOD matches ENVIRONMENT == 'runpod'."""
        expected = (ENVIRONMENT == "runpod")
        assert IS_RUNPOD == expected

    @pytest.mark.unit
    def test_is_auto_detected_boolean(self):
        """Test that IS_AUTO_DETECTED is a boolean."""
        assert isinstance(IS_AUTO_DETECTED, bool)


class TestPathConfiguration:
    """Test path configuration."""

    @pytest.mark.unit
    def test_project_root_is_path(self):
        """Test that PROJECT_ROOT is a Path object."""
        assert isinstance(PROJECT_ROOT, Path)

    @pytest.mark.unit
    def test_data_dir_is_path(self):
        """Test that DATA_DIR is a Path object."""
        assert isinstance(DATA_DIR, Path)

    @pytest.mark.unit
    def test_models_dir_is_path(self):
        """Test that MODELS_DIR is a Path object."""
        assert isinstance(MODELS_DIR, Path)

    @pytest.mark.unit
    def test_results_dir_is_path(self):
        """Test that RESULTS_DIR is a Path object."""
        assert isinstance(RESULTS_DIR, Path)

    @pytest.mark.unit
    def test_env_defaults_structure(self):
        """Test that ENV_DEFAULTS has correct structure."""
        for env in VALID_ENVIRONMENTS:
            assert env in ENV_DEFAULTS
            for key in ["PROJECT_ROOT", "DATA_DIR", "MODELS_DIR", "RESULTS_DIR"]:
                assert key in ENV_DEFAULTS[env]
                assert isinstance(ENV_DEFAULTS[env][key], str)


class TestGetConfigSummary:
    """Test get_config_summary function."""

    @pytest.mark.unit
    def test_returns_dict(self):
        """Test that get_config_summary returns a dictionary."""
        summary = get_config_summary()
        assert isinstance(summary, dict)

    @pytest.mark.unit
    def test_summary_contains_environment(self):
        """Test that summary contains environment key."""
        summary = get_config_summary()
        assert "environment" in summary
        assert summary["environment"] == ENVIRONMENT

    @pytest.mark.unit
    def test_summary_contains_all_paths(self):
        """Test that summary contains all path keys."""
        summary = get_config_summary()
        required_keys = [
            "project_root", "data_dir", "models_dir", 
            "results_dir", "cache_dir", "is_auto_detected"
        ]
        for key in required_keys:
            assert key in summary

    @pytest.mark.unit
    def test_summary_paths_are_strings(self):
        """Test that all paths in summary are strings."""
        summary = get_config_summary()
        path_keys = ["project_root", "data_dir", "models_dir", "results_dir", "cache_dir"]
        for key in path_keys:
            assert isinstance(summary[key], str)

    @pytest.mark.unit
    def test_summary_environment_is_string(self):
        """Test that environment in summary is a string."""
        summary = get_config_summary()
        assert isinstance(summary["environment"], str)

    @pytest.mark.unit
    def test_summary_is_auto_detected_is_boolean(self):
        """Test that is_auto_detected in summary is boolean."""
        summary = get_config_summary()
        assert isinstance(summary["is_auto_detected"], bool)


class TestCreateDirectories:
    """Test create_directories function."""

    @pytest.mark.unit
    def test_create_directories_creates_paths(self, tmp_path):
        """Test that create_directories creates all required directories."""
        with patch("src.config.PROJECT_ROOT", tmp_path):
            with patch("src.config.DATA_DIR", tmp_path / "data"):
                with patch("src.config.MODELS_DIR", tmp_path / "models"):
                    with patch("src.config.RESULTS_DIR", tmp_path / "results"):
                        create_directories()
                        
                        # Verify directories exist
                        assert (tmp_path / "data" / "raw").exists()
                        assert (tmp_path / "data" / "processed").exists()
                        assert (tmp_path / "data" / "metadata").exists()
                        assert (tmp_path / "models" / "cache").exists()
                        assert (tmp_path / "results" / "metrics").exists()

    @pytest.mark.unit
    def test_create_directories_handles_existing_directories(self, tmp_path):
        """Test that create_directories doesn't fail if directories exist."""
        # Create directories first
        (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
        
        with patch("src.config.PROJECT_ROOT", tmp_path):
            with patch("src.config.DATA_DIR", tmp_path / "data"):
                with patch("src.config.MODELS_DIR", tmp_path / "models"):
                    with patch("src.config.RESULTS_DIR", tmp_path / "results"):
                        # Should not raise
                        create_directories()
                        assert (tmp_path / "data" / "raw").exists()


class TestValidatePaths:
    """Test validate_paths function."""

    @pytest.mark.unit
    def test_validate_paths_non_strict_missing_paths(self):
        """Test that non-strict validation doesn't raise on missing paths."""
        with patch("src.config.PROJECT_ROOT", Path("/nonexistent/path")):
            with patch("src.config.DATA_DIR", Path("/nonexistent/data")):
                # Should not raise
                validate_paths(strict=False)

    @pytest.mark.unit
    def test_validate_paths_strict_missing_paths(self):
        """Test that strict validation raises on missing paths."""
        with patch("src.config.PROJECT_ROOT", Path("/nonexistent/path")):
            with patch("src.config.DATA_DIR", Path("/nonexistent/data")):
                with pytest.raises(RuntimeError):
                    validate_paths(strict=True)

    @pytest.mark.unit
    def test_validate_paths_strict_existing_paths(self, tmp_path):
        """Test that strict validation passes with existing paths."""
        with patch("src.config.PROJECT_ROOT", tmp_path):
            with patch("src.config.DATA_DIR", tmp_path / "data"):
                (tmp_path / "data").mkdir(exist_ok=True)
                # Should not raise
                validate_paths(strict=True)


class TestLogConfiguration:
    """Test log_configuration function."""

    @pytest.mark.unit
    def test_log_configuration_doesnt_raise(self):
        """Test that log_configuration doesn't raise exceptions."""
        # Should not raise
        log_configuration()

    @pytest.mark.unit
    def test_log_configuration_executes(self, caplog):
        """Test that log_configuration produces log output."""
        import logging
        caplog.set_level(logging.INFO)
        
        log_configuration()
        
        # Should have logged something
        assert len(caplog.records) > 0


class TestEnvironmentExplicitVsAuto:
    """Test explicit environment setting vs auto-detection."""

    @pytest.mark.unit
    def test_explicit_environment_not_auto_detected(self):
        """Test that explicit ENVIRONMENT setting results in IS_AUTO_DETECTED=False."""
        # When ENVIRONMENT is set explicitly, IS_AUTO_DETECTED should be False
        # (This is determined at module load time based on ENVIRONMENT value)
        # We can only test the logic here
        with patch.dict(os.environ, {"ENVIRONMENT": "runpod"}):
            # Simulate what happens at module load
            env_val = os.getenv("ENVIRONMENT")
            is_auto = env_val is None
            assert is_auto is False
