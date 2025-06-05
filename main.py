"""Backtest script for RSI + MACD strategy on Bybit.

This script downloads historical klines for selected symbols and
multiple timeframes from Bybit futures and applies a simple trading
strategy using RSI and MACD filters. It simulates positions with
10x leverage, $10 size, 3% take profit and 2% trailing stop.
It prints every trade and the final P/L.

Requirements (tested versions):
- requests==2.31.0
- pandas==1.5.3
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "pandas is required. Install with `pip install pandas==1.5.3`"
    ) from exc

import requests


# ====== Configurable parameters ======
BASE_URL: str = "https://api.bybit.com"
SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]
INTERVALS: List[str] = ["15", "30", "60", "240", "D"]
TAKE_PROFIT: float = 0.03  # 3%
TRAIL_STOP: float = 0.02  # 2%
LEVERAGE: float = 10.0
POSITION_SIZE: float = 10.0  # USD
RSI_PERIOD: int = 14


@dataclass
class Trade:
    """Structure describing a completed trade."""

    symbol: str
    open_time: datetime
    direction: str
    profit: float


def interval_seconds(interval: str) -> int:
    """Convert Bybit interval string to seconds."""
    if interval == "D":
        return 86400
    return int(interval) * 60


def fetch_klines(symbol: str, interval: str, start_ts: int) -> pd.DataFrame:
    """Download kline data from Bybit starting from `start_ts` in ms."""
    url = f"{BASE_URL}/v5/market/kline"
    params: Dict[str, str | int] = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "start": start_ts,
        "limit": 1000,
    }
    frames: List[Dict[str, float]] = []
    while True:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("retCode") != 0:
            break
        items = data["result"].get("list", [])
        if not items:
            break
        for candle in items:
            frames.append(
                {
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                }
            )
        last_ts = int(items[-1][0])
        next_ts = last_ts + interval_seconds(interval) * 1000
        if next_ts >= int(time.time() * 1000):
            break
        params["start"] = next_ts
        time.sleep(0.05)  # polite pause
    df = pd.DataFrame(frames)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series) -> pd.DataFrame:
    """Return MACD and signal line for a price series."""
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"MACD": macd, "Signal": signal})


def run_strategy(symbol: str, df: pd.DataFrame) -> List[Trade]:
    """Execute backtest on a DataFrame of klines."""
    df = df.copy()
    df["RSI"] = compute_rsi(df["close"])
    macd_df = compute_macd(df["close"])
    df = pd.concat([df, macd_df], axis=1)
    trades: List[Trade] = []
    position = None

    for idx in range(1, len(df)):
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        if position is None:
            # Look for entry points
            if row["RSI"] < 20 and prev["MACD"] < prev["Signal"] and row["MACD"] > row["Signal"]:
                position = {
                    "direction": "long",
                    "entry": row["close"],
                    "open_time": row["datetime"],
                    "peak": row["close"],
                }
            elif row["RSI"] > 80 and prev["MACD"] > prev["Signal"] and row["MACD"] < row["Signal"]:
                position = {
                    "direction": "short",
                    "entry": row["close"],
                    "open_time": row["datetime"],
                    "trough": row["close"],
                }
        else:
            if position["direction"] == "long":
                position["peak"] = max(position["peak"], row["high"])
                tp_price = position["entry"] * (1 + TAKE_PROFIT)
                trail_price = position["peak"] * (1 - TRAIL_STOP)
                if row["high"] >= tp_price:
                    exit_price = tp_price
                    profit = (exit_price - position["entry"]) / position["entry"] * LEVERAGE * POSITION_SIZE
                    trades.append(Trade(symbol, position["open_time"].to_pydatetime(), "long", profit))
                    position = None
                elif row["low"] <= trail_price:
                    exit_price = trail_price
                    profit = (exit_price - position["entry"]) / position["entry"] * LEVERAGE * POSITION_SIZE
                    trades.append(Trade(symbol, position["open_time"].to_pydatetime(), "long", profit))
                    position = None
            else:
                position["trough"] = min(position["trough"], row["low"])
                tp_price = position["entry"] * (1 - TAKE_PROFIT)
                trail_price = position["trough"] * (1 + TRAIL_STOP)
                if row["low"] <= tp_price:
                    exit_price = tp_price
                    profit = (position["entry"] - exit_price) / position["entry"] * LEVERAGE * POSITION_SIZE
                    trades.append(Trade(symbol, position["open_time"].to_pydatetime(), "short", profit))
                    position = None
                elif row["high"] >= trail_price:
                    exit_price = trail_price
                    profit = (position["entry"] - exit_price) / position["entry"] * LEVERAGE * POSITION_SIZE
                    trades.append(Trade(symbol, position["open_time"].to_pydatetime(), "short", profit))
                    position = None
    return trades


def main() -> None:
    """Run backtest for configured symbols and intervals."""
    start_ts = int((datetime.now(tz=timezone.utc) - timedelta(days=90)).timestamp() * 1000)
    all_trades: List[Trade] = []
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            try:
                df = fetch_klines(symbol, interval, start_ts)
            except Exception as exc:  # network or decoding errors
                print(f"Failed to load data for {symbol} {interval}: {exc}")
                continue
            trades = run_strategy(symbol, df)
            for trade in trades:
                print(
                    f"{trade.symbol} | {trade.open_time} | {trade.direction} | {trade.profit:.2f} USD"
                )
            all_trades.extend(trades)

    total_profit = sum(t.profit for t in all_trades)
    print(f"Total P/L: {total_profit:.2f} USD")


if __name__ == "__main__":
    main()
