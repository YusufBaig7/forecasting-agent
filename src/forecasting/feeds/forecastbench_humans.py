"""Helper functions for loading and aggregating ForecastBench human forecasts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd


def load_individual_forecasts(path: Path) -> pd.DataFrame:
    """
    Load a ForecastBench human_*_individual file into a DataFrame.
    
    Handles both JSONL and single JSON object with "forecasts" array.
    
    Returns columns: ["question_id", "user_id", "forecast", "source", "reasoning"].
    """
    path = Path(path)
    
    with path.open("r", encoding="utf-8") as f:
        # Check first non-whitespace character to detect format
        first_char = f.read(1)
        f.seek(0)
        
        if first_char.strip() == "{":
            # Single JSON object (likely has "forecasts" key)
            data = json.load(f)
            if "forecasts" in data:
                # Flatten the forecasts array
                rows = []
                for forecast in data["forecasts"]:
                    qid = forecast.get("id")
                    # Handle case where id might be a list (take first element)
                    if isinstance(qid, list):
                        if not qid:
                            continue
                        qid = qid[0]
                    if not isinstance(qid, str):
                        continue
                        
                    rows.append({
                        "question_id": qid,
                        "user_id": forecast.get("user_id"),
                        "forecast": forecast.get("forecast"),
                        "source": forecast.get("source"),
                        "reasoning": forecast.get("reasoning"),
                    })
                df = pd.DataFrame(rows)
            else:
                # Treat as single forecast object
                df = pd.DataFrame([{
                    "question_id": data.get("id"),
                    "user_id": data.get("user_id"),
                    "forecast": data.get("forecast"),
                    "source": data.get("source"),
                    "reasoning": data.get("reasoning"),
                }])
        elif first_char.strip() == "[":
            # Single JSON array
            df = pd.read_json(f)
            if "id" in df.columns:
                df = df.rename(columns={"id": "question_id"})
        else:
            # JSONL format
            df = pd.read_json(f, lines=True)
            if "id" in df.columns:
                df = df.rename(columns={"id": "question_id"})
    
    # Ensure required columns exist
    required_cols = ["question_id", "forecast"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing} in {path}")
    
    # Add optional columns if missing
    if "user_id" not in df.columns:
        df["user_id"] = None
    if "source" not in df.columns:
        df["source"] = None
    if "reasoning" not in df.columns:
        df["reasoning"] = None
    
    return df[["question_id", "user_id", "forecast", "source", "reasoning"]]


def aggregate_crowd(df: pd.DataFrame, agg: Literal["mean", "median"] = "median") -> pd.DataFrame:
    """
    Aggregate individual forecasts into a single crowd forecast per question.
    
    Args:
        df: DataFrame with columns ["question_id", "forecast", ...]
        agg: Aggregation method ("mean" or "median")
    
    Returns:
        DataFrame with columns ["question_id", "p_crowd"]
    """
    if "question_id" not in df.columns or "forecast" not in df.columns:
        raise ValueError("DataFrame must have 'question_id' and 'forecast' columns")
    
    # Group by question_id and aggregate
    if agg == "median":
        aggregated = df.groupby("question_id")["forecast"].median().reset_index()
    elif agg == "mean":
        aggregated = df.groupby("question_id")["forecast"].mean().reset_index()
    else:
        raise ValueError(f"Unknown aggregation method: {agg}. Use 'mean' or 'median'")
    
    aggregated = aggregated.rename(columns={"forecast": "p_crowd"})
    return aggregated[["question_id", "p_crowd"]]

