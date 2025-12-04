"""Forecast storage implementation."""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from forecasting.models import Forecast


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_git_hash() -> Optional[str]:
    """Get current git commit hash if available, else None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _generate_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())


class FileForecastStore:
    """
    Stores forecasts as JSONL files per run.
    
    Each run creates a single JSONL file with metadata header and forecast entries.
    Format:
    - First line: metadata JSON with run_id, git_hash, timestamp, model_name
    - Subsequent lines: one Forecast JSON per line
    """

    def __init__(self, root_dir: Path, run_id: Optional[str] = None):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or _generate_run_id()
        self.git_hash = _get_git_hash()
        self._file_handle = None
        self._file_path = None
        self._model_name = None
        self._run_started = False

    def _get_file_path(self, model_name: str) -> Path:
        """Get the JSONL file path for a model."""
        safe_model = model_name.replace("/", "_").replace(":", "_").replace("\\", "_")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{safe_model}_{self.run_id}_{timestamp}.jsonl"
        return self.root_dir / filename

    def start_run(self, model_name: str) -> Path:
        """
        Start a new forecast run. Must be called before logging forecasts.
        
        Returns the path to the JSONL file that will be created.
        """
        if self._run_started:
            raise RuntimeError("Run already started. Create a new FileForecastStore for a new run.")
        
        self._model_name = model_name
        self._file_path = self._get_file_path(model_name)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write metadata header
        metadata = {
            "run_id": self.run_id,
            "git_hash": self.git_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
        }
        
        with open(self._file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")
        
        self._run_started = True
        return self._file_path

    def log_forecast(self, forecast: Forecast) -> None:
        """
        Log a forecast to the current run's JSONL file.
        
        Must call start_run() first.
        """
        if not self._run_started:
            raise RuntimeError("Must call start_run() before logging forecasts.")
        
        if self._model_name != forecast.model:
            raise ValueError(
                f"Forecast model '{forecast.model}' doesn't match run model '{self._model_name}'"
            )
        
        # Append forecast as JSON line
        with open(self._file_path, "a", encoding="utf-8") as f:
            f.write(forecast.model_dump_json() + "\n")

    def read_run(self, file_path: Path) -> tuple[dict, list[Forecast]]:
        """
        Read a forecast run file and return (metadata, forecasts).
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            Tuple of (metadata dict, list of Forecast objects)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Forecast run file not found: {file_path}")
        
        metadata = None
        forecasts = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                if i == 0:
                    # First line is metadata
                    metadata = json.loads(line)
                else:
                    # Subsequent lines are forecasts
                    forecast_dict = json.loads(line)
                    forecasts.append(Forecast.model_validate(forecast_dict))
        
        if metadata is None:
            raise ValueError("Forecast run file missing metadata header")
        
        return metadata, forecasts

    def list_runs(self, model_name: Optional[str] = None) -> list[Path]:
        """
        List all forecast run files, optionally filtered by model name.
        
        Returns:
            List of paths to JSONL files
        """
        if not self.root_dir.exists():
            return []
        
        pattern = "*.jsonl"
        if model_name:
            safe_model = model_name.replace("/", "_").replace(":", "_").replace("\\", "_")
            pattern = f"{safe_model}_*.jsonl"
        
        return sorted(self.root_dir.glob(pattern), reverse=True)
