from typing import Iterable, List

ALL_MODELS_OPTION = "All models"


def expand_model_selection(
    selected_models: List[str],
    available_model_names: Iterable[str],
    all_option_label: str = ALL_MODELS_OPTION,
) -> List[str]:
    """Resolve a UI selection list into concrete model names.

    If the special all-option label is present, returns all available model names.
    Otherwise returns only selected items that exist in available_model_names,
    preserving order from available_model_names.
    """
    available = list(available_model_names)

    if all_option_label in selected_models:
        return available

    selected_set = set(selected_models)
    return [model_name for model_name in available if model_name in selected_set]


def normalize_model_multiselect(
    current_selection: List[str],
    previous_selection: List[str],
    all_option_label: str = ALL_MODELS_OPTION,
) -> List[str]:
    """Enforce mutual exclusivity for "All models" in a multiselect widget.

    Rules:
    - If user newly selects "All models", keep only that option.
    - If "All models" was selected and user then selects a specific model,
      remove "All models" and keep specific selections.
    """
    current = list(current_selection)

    if all_option_label in current and all_option_label not in previous_selection:
        return [all_option_label]

    if all_option_label in current and all_option_label in previous_selection and len(current) > 1:
        return [model_name for model_name in current if model_name != all_option_label]

    return current