"""Common utilities for the Lie Detection POC."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(file_path: Path) -> Any:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load CSV file."""
    return pd.read_csv(file_path)


def save_csv(df: pd.DataFrame, file_path: Path) -> None:
    """Save DataFrame to CSV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def save_numpy(array: np.ndarray, file_path: Path) -> None:
    """Save numpy array to file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)


def load_numpy(file_path: Path) -> np.ndarray:
    """Load numpy array from file."""
    return np.load(file_path)


def normalize_speaker_intention(intention: str) -> int:
    """Convert speaker_intention to binary label.
    
    Args:
        intention: "Lie" or "Truth"
    
    Returns:
        1 for Lie, 0 for Truth
    """
    if intention is None:
        return None
    intention_str = str(intention).lower()
    if intention_str == "lie":
        return 1
    elif intention_str == "truth":
        return 0
    else:
        logger.warning(f"Unknown intention value: {intention}")
        return None

