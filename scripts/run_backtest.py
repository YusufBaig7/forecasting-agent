"""Script to run backtests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from forecasting.config import get_settings
from forecasting.feeds.base import Feed
from forecasting.feeds.stub import StubFeed
from forecasting.forecast.baseline_market import MarketBaselineForecaster
from forecasting.forecast.llm_forecaster import LLMForecaster
from forecasting.forecast.multi_agent import MultiAgentForecaster
from forecasting.eval.backtest import run_backtest
from forecasting.llm.base import LLM
from forecasting.llm.mock import MockLLM
from forecasting.retrieval.base import Retriever
from forecasting.retrieval.stub import StubRetriever
from forecasting.storage.forecast_store import FileForecastStore
from forecasting.storage.snapshot_store import FileSnapshotStore

app = typer.Typer(add_completion=False)
console = Console()


def _get_llm(llm_type: str) -> LLM:
    """Get LLM instance based on type."""
    if llm_type == "mock":
        return MockLLM()
    elif llm_type == "gemini":
        try:
            from forecasting.llm.gemini_client import GeminiClientLLM
            return GeminiClientLLM()
        except (RuntimeError, ImportError) as e:
            console.print(f"[yellow]Warning: Gemini not available ({e}), falling back to mock[/yellow]")
            return MockLLM()
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def _get_retriever(retriever_type: str) -> Optional[Retriever]:
    """Get retriever instance based on type."""
    if retriever_type == "stub":
        return StubRetriever()
    elif retriever_type == "none":
        return None
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def _get_feed(feed_type: str, league: str = "nba") -> Feed:
    """Get feed instance based on type."""
    if feed_type == "stub":
        return StubFeed()
    elif feed_type == "sports":
        settings = get_settings()
        if not settings.sportsradar_api_key:
            raise ValueError(
                "SPORTSRADAR_API_KEY environment variable not set. "
                "Required for sports feed."
            )
        try:
            from forecasting.feeds.sports_odds import SportsOddsFeed
            return SportsOddsFeed(
                sportsradar_api_key=settings.sportsradar_api_key,
                sportsradar_base_url=settings.sportsradar_base_url,
                opticodds_api_key=settings.opticodds_api_key,
                opticodds_base_url=settings.opticodds_base_url,
                league=league,
            )
        except ImportError:
            raise ValueError(
                "Sports feed not available. Install httpx: pip install httpx"
            )
    else:
        raise ValueError(f"Unknown feed type: {feed_type}")


def _get_forecaster(
    forecaster_type: str,
    llm: Optional[LLM] = None,
    retriever: Optional[Retriever] = None,
    n_agents: int = 3,
    use_supervisor: bool = True,
) -> object:
    """Get forecaster instance based on type."""
    if forecaster_type == "market_baseline":
        return MarketBaselineForecaster()
    elif forecaster_type == "llm":
        if llm is None:
            raise ValueError("LLM required for llm forecaster")
        return LLMForecaster(llm=llm, retriever=retriever)
    elif forecaster_type == "multi_agent":
        if llm is None:
            raise ValueError("LLM required for multi_agent forecaster")
        return MultiAgentForecaster(
            llm=llm,
            retriever=retriever,
            n_agents=n_agents,
            use_supervisor=use_supervisor,
        )
    else:
        raise ValueError(f"Unknown forecaster type: {forecaster_type}")


@app.command()
def main(
    n_events: int = typer.Option(20, help="Number of stub events (for stub feed)"),
    n_asofs: int = typer.Option(4, help="Number of as_of timestamps"),
    hours_step: int = typer.Option(12, help="Hours between as_of timestamps"),
    feed: str = typer.Option("stub", help="Feed type: stub | sports"),
    league: str = typer.Option("nba", help="League code: nba | nfl | nhl | mlb (for sports feed)"),
    forecaster: str = typer.Option("market_baseline", help="Forecaster type: market_baseline | llm | multi_agent"),
    retriever: str = typer.Option("stub", help="Retriever type: stub | none"),
    llm: Optional[str] = typer.Option(None, help="LLM type: mock | gemini (auto-selects gemini if key exists)"),
    agents: int = typer.Option(3, help="Number of agents: 1 | 3 (for multi_agent)"),
    supervisor: str = typer.Option("on", help="Supervisor: on | off (for multi_agent)"),
    cache_snapshots: bool = typer.Option(True, help="Enable snapshot caching"),
    log_forecasts: bool = typer.Option(True, help="Enable forecast logging"),
    run_id: Optional[str] = typer.Option(None, help="Run ID (auto-generated if not provided)"),
):
    """Run a backtest with optional snapshot caching and forecast logging."""
    settings = get_settings()
    
    # Get feed
    try:
        if feed == "stub":
            feed_instance = StubFeed(n_events=n_events)
        elif feed == "sports":
            from forecasting.feeds.sports_odds import SportsOddsFeed
            feed_instance = SportsOddsFeed(
                settings.sportsradar_api_key,
                settings.sportsradar_base_url,
                settings.opticodds_api_key,
                settings.opticodds_base_url,
                league=league,
            )
        else:
            raise ValueError(f"Unknown feed: {feed}")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    # Auto-select LLM if not specified
    if llm is None:
        try:
            from forecasting.llm.gemini_client import get_llm_if_available
            llm_instance = get_llm_if_available()
            if llm_instance is not None:
                llm = "gemini"
                console.print("[dim]Auto-selected: gemini (API key found)[/dim]")
            else:
                llm = "mock"
                console.print("[dim]Auto-selected: mock (no API key)[/dim]")
        except ImportError:
            llm = "mock"
            console.print("[dim]Auto-selected: mock (Gemini not available)[/dim]")
    
    # Get LLM and retriever if needed
    llm_instance = None
    retriever_instance = None
    if forecaster in ("llm", "multi_agent"):
        llm_instance = _get_llm(llm)
        retriever_instance = _get_retriever(retriever)
    
    # Parse supervisor flag
    use_supervisor = supervisor.lower() == "on"
    
    # Validate agents count
    if agents not in (1, 3):
        raise ValueError("agents must be 1 or 3")
    
    forecaster_instance = _get_forecaster(
        forecaster,
        llm_instance,
        retriever_instance,
        n_agents=agents,
        use_supervisor=use_supervisor,
    )

    now = datetime.now(timezone.utc)
    as_ofs = [now - timedelta(hours=hours_step * i) for i in range(n_asofs)][::-1]

    snapshot_store = None
    if cache_snapshots:
        snapshot_store = FileSnapshotStore(settings.snapshots_dir)
        console.print(f"[dim]Snapshot cache: {settings.snapshots_dir}[/dim]")

    forecast_store = None
    if log_forecasts:
        forecast_store = FileForecastStore(settings.forecasts_dir, run_id=run_id)
        console.print(f"[dim]Forecast log: {settings.forecasts_dir}[/dim]")
        console.print(f"[dim]Run ID: {forecast_store.run_id}[/dim]")

    result = run_backtest(
        feed=feed_instance,
        forecaster=forecaster_instance,
        as_of_times=as_ofs,
        snapshot_store=snapshot_store,
        forecast_store=forecast_store,
    )
    
    if forecast_store is not None:
        # Get the file path that was created
        run_file = forecast_store._file_path
        console.print(f"[dim]Forecasts logged to: {run_file}[/dim]")

    console.print(f"[bold]Brier:[/bold] {result.brier:.6f}")
    console.print(f"[bold]ECE:[/bold]   {result.ece:.6f}")
    
    if result.trading is not None:
        console.print(f"[bold]Return:[/bold]      {result.trading.total_return:.4f} ({result.trading.total_return*100:.2f}%)")
        console.print(f"[bold]Sharpe:[/bold]      {result.trading.sharpe:.4f}")
        console.print(f"[bold]Max Drawdown:[/bold] {result.trading.max_drawdown:.4f} ({result.trading.max_drawdown*100:.2f}%)")
        console.print(f"[bold]Bets:[/bold]        {result.trading.num_bets}")
    else:
        console.print("[dim]Trading metrics: N/A (no market probabilities available)[/dim]")

    df = result.predictions.sort_values(["as_of", "event_id"]).head(12)

    table = Table(title="Sample predictions (first 12 rows)")
    for col in ["as_of", "event_id", "p_yes", "y", "q_market", f"abs(p_yes - q_market)"]:
        table.add_column(col)

    for _, r in df.iterrows():
        y = r["y"]
        q = r.get("q_market", None)

        if q is None:
            q_str = "NA"
            abs_str = "NA"
        else:
            q = float(q)
            q_str = f"{q:.3f}"
            abs_str = f"{abs(float(r['p_yes']) - q):.3f}"

        table.add_row(
            str(r["as_of"]),
            str(r["event_id"]),
            f"{float(r['p_yes']):.3f}",
            str(int(y)) if y is not None else "NA",
            q_str,
            abs_str,
        )

    console.print(table)


if __name__ == "__main__":
    app()
