"""
Classifier Registry - Central source of truth for classifier types.

This module defines supported LLM classifiers and provides utilities
to distinguish between LLM classifiers and file-based classifiers.
"""

from pathlib import Path
from typing import Union

# Supported LLM model names that can be used as classifiers
# These models will use prompt engineering to classify scene boundaries
SUPPORTED_LLM_CLASSIFIERS = {
    # OpenAI GPT models
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",

    # OpenAI o-series (reasoning models)
    "o1",
    "o1-mini",
    "o1-preview",
    "o3-mini",

    # Add more LLM providers as needed
    # "claude-3-opus",
    # "claude-3-sonnet",
}


def is_llm_classifier(identifier: str) -> bool:
    """
    Check if the given identifier is a supported LLM classifier.

    Args:
        identifier: Either an LLM model name or a file path

    Returns:
        True if the identifier is a registered LLM classifier name
    """
    if not isinstance(identifier, str):
        return False

    return identifier in SUPPORTED_LLM_CLASSIFIERS


def get_classifier_type(classifier_path: Union[str, None]) -> str:
    """
    Determine the type of classifier from its identifier.

    Args:
        classifier_path: Either an LLM name, file path (directory or file), or None

    Returns:
        One of: 'llm', 'llm-finetune', 'pytorch', 'xgboost', or 'none'

    Raises:
        ValueError: If the classifier type cannot be determined
    """
    if classifier_path is None:
        return 'none'

    if is_llm_classifier(classifier_path):
        return 'llm'

    # Check if it's a file or directory path
    path = Path(classifier_path)

    if not path.exists():
        # Provide helpful error message
        raise ValueError(
            f"Classifier '{classifier_path}' not found.\n"
            f"Supported LLM names: {sorted(SUPPORTED_LLM_CLASSIFIERS)}\n"
            f"Or provide a valid path (.pth for PyTorch, .json for XGBoost, directory for fine-tuned LLM)"
        )

    # Check if it's a directory-based classifier
    if path.is_dir():
        config_file = path / 'config.json'

        # Check for fine-tuned LLM (has adapter_config.json or config.json)
        if (path / 'adapter_config.json').exists() or config_file.exists():
            return 'llm-finetune'
        else:
            raise ValueError(
                f"Directory '{classifier_path}' doesn't appear to be a valid model.\n"
                f"Expected config files for fine-tuned LLM model inside."
            )

    # File-based classifiers
    if path.suffix == '.pth':
        return 'pytorch'
    elif path.suffix == '.json':
        return 'xgboost'
    else:
        raise ValueError(
            f"Unknown classifier file format: {path.suffix}\n"
            f"Supported formats: .pth (PyTorch), .json (XGBoost), directory (fine-tuned LLM)\n"
            f"Or use an LLM name: {sorted(SUPPORTED_LLM_CLASSIFIERS)}"
        )


def validate_classifier(classifier_path: Union[str, None]) -> None:
    """
    Validate that a classifier identifier is valid.

    Args:
        classifier_path: Either an LLM name, file path, or None

    Raises:
        ValueError: If the classifier is invalid
    """
    get_classifier_type(classifier_path)  # Will raise if invalid
