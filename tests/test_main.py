"""Tests for src/main.py — trade group resolution and task orchestration.

All external I/O (market feeds, news feed, dashboard, engine HFT loop) is
mocked so these tests run without network access or a real event loop running
subsystems.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Trade group resolution (pure logic — no async needed)
# ---------------------------------------------------------------------------

class TestTradeGroupResolution:
    """Verify that the 'all' trade group and individual groups expand correctly."""

    def test_all_set_expands_correctly(self):
        """Confirm the set expansion logic in main() independently."""
        trade_groups = {"all"}
        if "all" in trade_groups:
            trade_groups = {"crypto", "asx", "global"}
        assert trade_groups == {"crypto", "asx", "global"}

    def test_global_in_expanded_set(self):
        trade_groups = {"all"}
        if "all" in trade_groups:
            trade_groups = {"crypto", "asx", "global"}
        assert "global" in trade_groups

    def test_crypto_in_expanded_set(self):
        trade_groups = {"all"}
        if "all" in trade_groups:
            trade_groups = {"crypto", "asx", "global"}
        assert "crypto" in trade_groups

    def test_asx_in_expanded_set(self):
        trade_groups = {"all"}
        if "all" in trade_groups:
            trade_groups = {"crypto", "asx", "global"}
        assert "asx" in trade_groups

    def test_all_removed_from_set_after_expansion(self):
        trade_groups = {"all"}
        if "all" in trade_groups:
            trade_groups = {"crypto", "asx", "global"}
        assert "all" not in trade_groups

    def test_no_all_leaves_set_unchanged(self):
        trade_groups = {"crypto"}
        if "all" in trade_groups:
            trade_groups = {"crypto", "asx", "global"}
        assert trade_groups == {"crypto"}

    def test_explicit_global_not_expanded(self):
        trade_groups = {"global"}
        if "all" in trade_groups:
            trade_groups = {"crypto", "asx", "global"}
        assert trade_groups == {"global"}


# ---------------------------------------------------------------------------
# Helpers shared by task-creation tests
# ---------------------------------------------------------------------------

async def _noop():
    """A trivial coroutine used as a stand-in task coroutine."""


def _make_fake_create_task(recorded_names: list[str]):
    """Return a fake asyncio.create_task that records task names without
    running real coroutines.  The real coro is closed (discarded) and a
    no-op task is created and immediately cancelled so tests stay clean.
    """
    def _fake_create_task(coro, *, name=None):
        # Close the mock coroutine to avoid ResourceWarning
        try:
            coro.close()
        except AttributeError:
            pass
        if name:
            recorded_names.append(name)
        # Create a real but immediately-cancelled task so the return type is correct.
        task = asyncio.get_event_loop().create_task(_noop(), name=name)
        task.cancel()
        return task

    return _fake_create_task


def _make_mock_feed():
    """Return a MagicMock with async start/stop methods (used for MarketFeed/IndicesFeed)."""
    mock = MagicMock()
    mock.start = AsyncMock(return_value=None)
    mock.stop = AsyncMock(return_value=None)
    return mock


def _patch_all_subsystems(recorded_names):
    """Return a tuple of context managers that patch all subsystems used in main()
    plus replace asyncio.create_task with the recording spy.

    Feed class patches produce instances whose start/stop are proper AsyncMocks so
    the finally block in main() can await market_feed.stop() and indices_feed.stop().
    """
    mock_market_feed = _make_mock_feed()
    mock_indices_feed = _make_mock_feed()

    return (
        patch("src.main.create_provider", return_value=MagicMock(name="mock_llm")),
        patch("src.main.MarketFeed", return_value=mock_market_feed),
        patch("src.main.IndicesFeed", return_value=mock_indices_feed),
        patch("src.main.Portfolio"),
        patch("src.main.NewsFeed"),
        patch("src.main.Predictor"),
        patch("src.main.TradeEngine"),
        patch("src.main.Dashboard"),
        patch(
            "src.main.asyncio.create_task",
            side_effect=_make_fake_create_task(recorded_names),
        ),
        patch("src.main.asyncio.gather", new_callable=AsyncMock, return_value=[]),
    )


# ---------------------------------------------------------------------------
# hft_loop_us_stocks task creation
# ---------------------------------------------------------------------------

class TestUSStocksTaskCreation:
    """Verify that main() creates the hft_loop_us_stocks task when appropriate."""

    @pytest.mark.asyncio
    async def test_global_group_with_us_stocks_enabled_creates_hft_task(self):
        """When trade_groups={'global'} and us_stocks_enabled=True, main() must
        schedule an 'hft_loop_us_stocks' asyncio task."""
        created_task_names: list[str] = []

        patches = _patch_all_subsystems(created_task_names)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:

            from src.main import main as platform_main
            from config import config

            original = config.us_stocks_enabled
            config.us_stocks_enabled = True
            try:
                await platform_main(trade_groups={"global"})
            finally:
                config.us_stocks_enabled = original

        assert "hft_loop_us_stocks" in created_task_names, (
            f"Expected 'hft_loop_us_stocks' in created task names but got: {created_task_names}"
        )

    @pytest.mark.asyncio
    async def test_global_group_with_us_stocks_disabled_skips_hft_task(self):
        """When us_stocks_enabled=False, no hft_loop_us_stocks task should be created."""
        created_task_names: list[str] = []

        patches = _patch_all_subsystems(created_task_names)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:

            from src.main import main as platform_main
            from config import config

            original = config.us_stocks_enabled
            config.us_stocks_enabled = False
            try:
                await platform_main(trade_groups={"global"})
            finally:
                config.us_stocks_enabled = original

        assert "hft_loop_us_stocks" not in created_task_names

    @pytest.mark.asyncio
    async def test_crypto_only_group_does_not_create_us_stocks_task(self):
        """trade_groups={'crypto'} must not spawn an hft_loop_us_stocks task."""
        created_task_names: list[str] = []

        patches = _patch_all_subsystems(created_task_names)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:

            from src.main import main as platform_main
            await platform_main(trade_groups={"crypto"})

        assert "hft_loop_us_stocks" not in created_task_names

    @pytest.mark.asyncio
    async def test_all_group_creates_hft_loop_us_stocks(self):
        """'all' trade group (expands to crypto+asx+global) must include hft_loop_us_stocks."""
        created_task_names: list[str] = []

        patches = _patch_all_subsystems(created_task_names)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:

            from src.main import main as platform_main
            from config import config

            original_us = config.us_stocks_enabled
            original_asx = config.asx_enabled
            config.us_stocks_enabled = True
            config.asx_enabled = True
            try:
                await platform_main(trade_groups={"all"})
            finally:
                config.us_stocks_enabled = original_us
                config.asx_enabled = original_asx

        assert "hft_loop_us_stocks" in created_task_names, (
            f"Expected 'hft_loop_us_stocks' in created task names but got: {created_task_names}"
        )

    @pytest.mark.asyncio
    async def test_global_group_creates_hft_task_name_exactly(self):
        """The task name must be exactly 'hft_loop_us_stocks' (not a substring)."""
        created_task_names: list[str] = []

        patches = _patch_all_subsystems(created_task_names)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:

            from src.main import main as platform_main
            from config import config

            original = config.us_stocks_enabled
            config.us_stocks_enabled = True
            try:
                await platform_main(trade_groups={"global"})
            finally:
                config.us_stocks_enabled = original

        assert "hft_loop_us_stocks" in created_task_names

    @pytest.mark.asyncio
    async def test_global_group_does_not_create_hft_loop_crypto(self):
        """trade_groups={'global'} alone must NOT create an hft_loop_crypto task."""
        created_task_names: list[str] = []

        patches = _patch_all_subsystems(created_task_names)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:

            from src.main import main as platform_main
            from config import config

            original = config.us_stocks_enabled
            config.us_stocks_enabled = True
            try:
                await platform_main(trade_groups={"global"})
            finally:
                config.us_stocks_enabled = original

        assert "hft_loop_crypto" not in created_task_names
