"""Metrics logging utilities for RL training.

This module provides centralized logging functionality for RL algorithms,
including CSV export, TensorBoard integration, and WandB support.
"""

import csv
import os
from typing import Any, Dict, List


def save_metrics_history(save_path: str, metrics_history: List[Dict[str, Any]]) -> None:
    """Write aggregated metrics to a CSV file for readability.

    Args:
        save_path: Path to save the CSV file
        metrics_history: List of dictionaries containing metrics for each iteration

    The function automatically collects all unique keys from the metrics history
    and creates a CSV with dynamic columns based on the logged metrics.
    """
    if not metrics_history:
        return

    # Collect all unique fieldnames across all entries
    fieldnames: List[str] = []
    for entry in metrics_history:
        for key in entry.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Write CSV file
    with open(save_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_history)

    print(f"Saved metrics history to {save_path}")
