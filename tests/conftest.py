"""Shared fixtures for all tests."""
import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_prices():
    """200 realistic BTC prices for indicator tests."""
    import numpy as np
    np.random.seed(42)
    base = 45000.0
    # Random walk
    changes = np.random.randn(200) * 200
    prices = [base + changes[:i + 1].sum() for i in range(200)]
    return [float(p) for p in prices]


@pytest.fixture
def mock_config():
    from config import AppConfig
    return AppConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        initial_cash=10_000.0,
        hft_interval=0.1,
        trade_fraction=0.05,
        max_position_fraction=0.2,
        claude_api_key="test-key",
    )
