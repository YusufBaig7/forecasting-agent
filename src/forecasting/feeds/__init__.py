"""Data feed modules."""

from .base import Feed
from .stub import StubFeed

__all__ = ["Feed", "StubFeed"]

# Conditionally export sports feed if available
try:
    from .sports_odds import SportsOddsFeed
    __all__.append("SportsOddsFeed")
except ImportError:
    pass

# Export ForecastBench feed
try:
    from .forecastbench_feed import ForecastBenchFeed
    __all__.append("ForecastBenchFeed")
except ImportError:
    pass


