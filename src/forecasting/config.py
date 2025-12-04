"""Configuration management for the forecasting agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "data")).resolve()
    snapshots_dir: Path = Path(os.getenv("SNAPSHOTS_DIR", "data/snapshots")).resolve()
    forecasts_dir: Path = Path(os.getenv("FORECASTS_DIR", "data/forecasts")).resolve()
    
    # SportsRadar API settings
    sportsradar_api_key: Optional[str] = os.getenv("SPORTSRADAR_API_KEY")
    sportsradar_base_url: str = os.getenv("SPORTSRADAR_BASE_URL", "https://api.sportradar.com")
    
    # OpticOdds API settings
    opticodds_api_key: Optional[str] = os.getenv("OPTICODDS_API_KEY")
    opticodds_base_url: str = os.getenv("OPTICODDS_BASE_URL", "https://api.opticodds.com")


def get_settings() -> Settings:
    return Settings()
