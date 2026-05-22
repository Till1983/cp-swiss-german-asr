"""Unit tests for model selection helpers."""

from src.frontend.utils.model_selection import (
    ALL_MODELS_OPTION,
    expand_model_selection,
    normalize_model_multiselect,
)


class TestExpandModelSelection:
    """Tests for expand_model_selection function."""

    def test_returns_all_models_when_all_option_selected(self):
        available_models = ["model-a", "model-b", "model-c"]

        selected = expand_model_selection(
            selected_models=[ALL_MODELS_OPTION],
            available_model_names=available_models,
        )

        assert selected == available_models

    def test_returns_all_models_when_all_option_and_specific_models_selected(self):
        available_models = ["model-a", "model-b", "model-c"]

        selected = expand_model_selection(
            selected_models=[ALL_MODELS_OPTION, "model-a"],
            available_model_names=available_models,
        )

        assert selected == available_models

    def test_filters_out_unknown_models(self):
        available_models = ["model-a", "model-b"]

        selected = expand_model_selection(
            selected_models=["model-a", "missing-model"],
            available_model_names=available_models,
        )

        assert selected == ["model-a"]

    def test_preserves_available_model_order(self):
        available_models = ["model-b", "model-a", "model-c"]

        selected = expand_model_selection(
            selected_models=["model-a", "model-c"],
            available_model_names=available_models,
        )

        assert selected == ["model-a", "model-c"]

    def test_returns_empty_when_nothing_selected(self):
        available_models = ["model-a", "model-b"]

        selected = expand_model_selection(
            selected_models=[],
            available_model_names=available_models,
        )

        assert selected == []


class TestNormalizeModelMultiselect:
    """Tests for normalize_model_multiselect function."""

    def test_newly_selecting_all_models_keeps_only_all(self):
        normalized = normalize_model_multiselect(
            current_selection=["model-a", ALL_MODELS_OPTION],
            previous_selection=["model-a"],
        )

        assert normalized == [ALL_MODELS_OPTION]

    def test_selecting_specific_model_after_all_removes_all(self):
        normalized = normalize_model_multiselect(
            current_selection=[ALL_MODELS_OPTION, "model-b"],
            previous_selection=[ALL_MODELS_OPTION],
        )

        assert normalized == ["model-b"]

    def test_leaves_specific_models_unchanged(self):
        normalized = normalize_model_multiselect(
            current_selection=["model-a", "model-b"],
            previous_selection=["model-a"],
        )

        assert normalized == ["model-a", "model-b"]
