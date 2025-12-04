# forecasting-agent

A modular forecasting system for binary prediction markets with time-bounded evaluation, multiple forecaster types, and comprehensive metrics.

## Overview

This system provides a complete pipeline for:
- **Fetching market data** from various sources (synthetic, sports, ForecastBench)
- **Generating forecasts** using market baselines, LLMs, or multi-agent systems
- **Evaluating predictions** with proper time-bounded evaluation (Brier score, ECE, trading metrics)
- **Comparing against human baselines** on ForecastBench datasets

All operations are **time-bounded** via `as_of` timestamps to prevent data leakage and ensure realistic evaluation.

## Quick Start

```bash
# Install
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# Run a backtest with synthetic data
python scripts/run_backtest.py --n-events 30 --n-asofs 5

# Run experiments across multiple configurations
python scripts/run_experiments.py --n-events 30 --n-asofs 5

# Evaluate on ForecastBench
python scripts/run_forecastbench.py --use-llm --use-multi-agent
```

## Core Concepts

### Data Models

- **`Event`**: Forecasting question with metadata (title, question, close time, resolution criteria)
- **`MarketSnapshot`**: Market state at a point in time (probabilities, odds, liquidity)
- **`Forecast`**: Model prediction with probability, rationale, and metadata
- **`ResolvedOutcome`**: Binary outcome (0=NO, 1=YES) for resolved events

### Time-Bounded Evaluation

All operations use `as_of` timestamps to ensure:
- No future information leaks into past predictions
- Realistic evaluation that matches production conditions
- Reproducible backtests with proper temporal ordering

## Features

### Data Feeds

**StubFeed** (`src/forecasting/feeds/stub.py`)
- Deterministic synthetic data for development/testing
- Generates stable event lists and market snapshots
- Useful for testing and development

**SportsOddsFeed** (`src/forecasting/feeds/sports_odds.py`)
- Real sports data from SportsRadar (schedules, outcomes)
- Moneyline odds from OpticOdds (converted to vig-free probabilities)
- Supports NBA, NFL, NHL, MLB (NBA currently implemented)
- Binary market: YES = home team wins, NO = home team does not win
- Automatic fixture matching and sportsbook selection
- Historical odds support (rolling ~2-month retention)

**ForecastBenchFeed** (`src/forecasting/feeds/forecastbench_feed.py`)
- ForecastBench (FB-Market style) dataset support
- Standard question sets or human forecast files (fallback)
- Source filtering (prediction markets only or all sources)

### Forecasters

**MarketBaselineForecaster**
- Uses market-implied probabilities directly
- Simple baseline for comparison

**LLMForecaster**
- Single LLM-based forecaster
- Supports provider-agnostic LLM interface (Mock, OpenAI, Gemini)
- Optional retrieval for context

**MultiAgentForecaster**
- Multiple specialized agents (Market, News, Skeptic)
- Optional supervisor for disagreement handling
- Combines agent perspectives with confidence weighting

**CalibratedForecaster**
- Wraps any forecaster with calibration
- Platt scaling for probability calibration
- Optional extremizer for probability adjustment

### Evaluation Metrics

- **Brier Score**: Prediction accuracy (lower is better, 0 = perfect)
- **Expected Calibration Error (ECE)**: Calibration quality (lower is better, 0 = perfectly calibrated)
- **Negative Log Likelihood (NLL)**: Probabilistic scoring
- **Trading Metrics**: Return, Sharpe ratio, max drawdown (Kelly sizing strategy)
- **Coverage**: Fraction of possible forecasts produced

### Storage

- **FileSnapshotStore**: Caches market snapshots as JSON files
- **FileForecastStore**: Logs forecasts as JSONL files with metadata
- Automatic caching and logging integrated into backtests

## CLI Tools

### `run_backtest.py`

Standard backtest with configurable parameters.

```bash
# Basic usage
python scripts/run_backtest.py --n-events 30 --n-asofs 5

# With sports feed
python scripts/run_backtest.py --feed sports --league nba --forecaster market_baseline

# With LLM forecaster
python scripts/run_backtest.py --forecaster llm --use-llm

# Options
--feed stub|sports              # Feed type (default: stub)
--league nba|nfl|nhl|mlb        # League for sports feed (default: nba)
--forecaster market_baseline|llm|multi_agent
--n-events N                    # Number of stub events (default: 20)
--n-asofs N                     # Number of as_of timestamps (default: 4)
--hours-step N                  # Hours between timestamps (default: 12)
--cache-snapshots               # Enable snapshot caching (default: enabled)
--log-forecasts                 # Enable forecast logging (default: enabled)
--run-id ID                     # Custom run ID (auto-generated if not provided)
```

### `run_experiments.py`

Run multiple forecasting configurations and generate a comprehensive report.

```bash
python scripts/run_experiments.py --n-events 30 --n-asofs 5

# Outputs:
# - data/reports/{run_id}.json (machine-readable)
# - data/reports/{run_id}.md (human-readable table)
```

Evaluates: Market baseline, LLM, multi-agent (with/without supervisor), and calibrated variants.

### `run_forecastbench.py`

Evaluate forecasters on ForecastBench datasets and compare against human baselines.

```bash
# Basic evaluation
python scripts/run_forecastbench.py

# With LLM and multi-agent
python scripts/run_forecastbench.py --use-llm --use-multi-agent

# With calibration
python scripts/run_forecastbench.py --use-llm --calibrator-path data/calibration/my_calibrator.json

# Options
--question-set PATH              # Question set JSON (default: data/forecastbench/question_sets/2024-07-21-human.json)
--resolution-set PATH           # Resolution set JSON
--public-forecasts PATH         # Public crowd forecasts JSON
--super-forecasts PATH          # Superforecaster forecasts JSON
--split fb_market|all           # Source filter (default: fb_market)
--use-llm                       # Include LLM forecaster
--use-multi-agent               # Include multi-agent forecaster
--calibrator-path PATH          # Path to saved calibrator
```

**Output**: Table comparing automated forecasters against human baselines (Public Crowd, Superforecasters) with Brier, ECE, NLL, and coverage metrics.

### `run_historical_sports_backtest.py`

Comprehensive historical sports backtesting across date ranges.

```bash
python scripts/run_historical_sports_backtest.py \
    --league nba \
    --forecaster market_baseline \
    --start-date 2024-01-01 \
    --end-date 2024-01-31

# Generates forecasts at multiple timestamps per game:
# - open: ~24h before kickoff
# - day_of: ~6h before kickoff
# - 1h: 1 hour before kickoff
# Outputs CSV: data/forecasts/historical_{league}_{model}.csv
```

### `fit_calibrator.py`

Train probability calibration models on historical predictions.

```bash
python scripts/fit_calibrator.py \
    --n-events 50 \
    --n-asofs 10 \
    --train-split 0.7 \
    --name my_calibrator

# Saves to: data/calibration/my_calibrator.json
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Storage directories
DATA_DIR=data
SNAPSHOTS_DIR=data/snapshots
FORECASTS_DIR=data/forecasts

# LLM (optional)
GEMINI_API_KEY=your_key_here          # Uses MockLLM if not set

# Sports feed (required for --feed sports)
SPORTSRADAR_API_KEY=your_key_here     # Required
SPORTSRADAR_BASE_URL=https://api.sportradar.com  # Optional

# OpticOdds (optional but recommended)
OPTICODDS_API_KEY=your_key_here       # Needed for market snapshots
OPTICODDS_BASE_URL=https://api.opticodds.com     # Optional
OPTICODDS_SPORTSBOOK=book1,book2      # Pin specific sportsbooks (optional)
```

### ForecastBench Data

Place ForecastBench files under `data/forecastbench/`:

```
data/forecastbench/
├── question_sets/
│   ├── 2024-07-21-human.json                    # Question set (required)
│   ├── 2024-07-21.ForecastBench.human_public_individual.json    # Optional
│   └── 2024-07-21.ForecastBench.human_super_individual.json    # Optional
└── resolution_sets/
    └── 2024-07-21_resolution_set.json           # Resolution set (required)
```

## Examples

### Basic Backtest

```bash
# Synthetic data
python scripts/run_backtest.py --n-events 30 --n-asofs 5

# Sports data (requires API keys)
python scripts/run_backtest.py --feed sports --league nba --n-asofs 1
```

### Multi-Agent Forecasting

```bash
# Run with multi-agent system (uses MockLLM if no API key)
python scripts/run_backtest.py --forecaster multi_agent --agents 3 --supervisor on
```

### Calibration Workflow

```bash
# 1. Fit calibrator on historical data
python scripts/fit_calibrator.py --n-events 50 --n-asofs 10 --name my_calibrator

# 2. Use calibrated forecaster
python scripts/run_backtest.py \
    --forecaster market_baseline \
    --calibrator-path data/calibration/my_calibrator.json
```

### ForecastBench Evaluation

```bash
# Compare all forecasters against human baselines
python scripts/run_forecastbench.py \
    --use-llm \
    --use-multi-agent \
    --calibrator-path data/calibration/my_calibrator.json
```

## Architecture

### Design Principles

- **Time-bounded evaluation**: All operations use `as_of` timestamps
- **Modular design**: Clean separation between feeds, storage, forecasting, and evaluation
- **Extensible**: Easy to add new feeds, forecasters, or metrics
- **Type-safe**: Uses Pydantic models for validation
- **UTC normalization**: All datetimes normalized to UTC

### Key Components

```
src/forecasting/
├── models.py              # Core data models (Event, MarketSnapshot, Forecast, ResolvedOutcome)
├── feeds/                 # Data feed implementations
│   ├── base.py            # Feed interface
│   ├── stub.py            # Synthetic data
│   ├── sports_odds.py     # Sports data (SportsRadar + OpticOdds)
│   └── forecastbench_feed.py  # ForecastBench dataset
├── forecast/              # Forecaster implementations
│   ├── baseline_market.py
│   ├── llm_forecaster.py
│   ├── multi_agent.py
│   ├── agents.py
│   ├── supervisor.py
│   └── calibration.py
├── eval/                  # Evaluation metrics and backtesting
│   ├── metrics.py         # Brier score, ECE
│   ├── trading.py         # Trading simulator
│   └── backtest.py        # Backtest runner
├── storage/               # Persistence
│   ├── snapshot_store.py  # Market snapshot caching
│   └── forecast_store.py  # Forecast logging
├── llm/                   # LLM providers
│   ├── base.py            # LLM interface
│   ├── mock.py            # Deterministic mock LLM
│   ├── openai_client.py   # OpenAI client
│   └── gemini_client.py   # Gemini client
└── retrieval/             # Context retrieval
    ├── base.py            # Retriever interface
    └── stub.py            # Stub retriever
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Test coverage includes:
- Storage modules (snapshot and forecast stores)
- Evaluation metrics (Brier score, ECE)
- Trading simulator
- Calibration (Platt scaling, extremizer)
- Sports feed (with mocked HTTP responses)
- ForecastBench feed
- Multi-agent system
- Time-bounded operations

## Data Storage

### Snapshot Cache

- **Location**: `data/snapshots/{event_id}/{YYYYMMDDTHHMMSSZ}.json`
- **Behavior**: Automatically caches snapshots during backtests
- **Benefit**: Speeds up repeated runs and enables reproducible backtests

### Forecast Logs

- **Location**: `data/forecasts/{model}_{run_id}_{timestamp}.jsonl`
- **Format**: Metadata JSON on first line, one Forecast JSON per subsequent line
- **Usage**: Track all predictions for analysis and debugging

### Reports

- **Location**: `data/reports/{run_id}.json` and `{run_id}.md`
- **Content**: Experiment results with metrics tables

## License

[Add license information if applicable]
