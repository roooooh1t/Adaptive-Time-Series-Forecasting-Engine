"""
Loads raw M5 CSVs from disk into pandas DataFrames.
No cleaning or transformation happens here — purely I/O.
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sales(raw_dir: str, evaluation: bool = False) -> pd.DataFrame:
    """
    Load the wide-format sales file.

    Parameters
    ----------
    evaluation : bool
        M5 ships two versions — sales_train_validation.csv (28 fewer days)
        and sales_train_evaluation.csv (full history, used for final scoring).
        Use evaluation=True once you're doing final benchmarking, not during
        early development.
    """
    filename = "sales_train_evaluation.csv" if evaluation else "sales_train_validation.csv"
    path = Path(raw_dir) / filename
    df = pd.read_csv(path)
    print(f"[ingest] Loaded sales: {df.shape[0]:,} rows x {df.shape[1]:,} cols from {filename}")
    return df


def load_calendar(raw_dir: str) -> pd.DataFrame:
    path = Path(raw_dir) / "calendar.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"[ingest] Loaded calendar: {df.shape[0]:,} rows")
    return df


def load_prices(raw_dir: str) -> pd.DataFrame:
    path = Path(raw_dir) / "sell_prices.csv"
    df = pd.read_csv(path)
    print(f"[ingest] Loaded prices: {df.shape[0]:,} rows")
    return df


def load_all(raw_dir: str, evaluation: bool = False) -> dict:
    """Convenience wrapper — loads all three files and returns as a dict."""
    return {
        "sales": load_sales(raw_dir, evaluation=evaluation),
        "calendar": load_calendar(raw_dir),
        "prices": load_prices(raw_dir),
    }


if __name__ == "__main__":
    config = load_config()
    raw_dir = config["paths"]["raw_data"]
    data = load_all(raw_dir)
    for name, df in data.items():
        print(f"{name}: {df.shape}")