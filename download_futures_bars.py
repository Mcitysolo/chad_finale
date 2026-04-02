#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT_DIR = Path("/home/ubuntu/chad_finale/data/bars/1d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL_MAP = {
    "MES": "ES=F",
    "MNQ": "NQ=F",
    "MCL": "CL=F",
    "MGC": "GC=F",
}


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dataframe is flat and usable
    """
    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    return df


def fetch(symbol, yf_symbol):
    print(f"Downloading {symbol} ({yf_symbol})...")

    df = yf.download(
        yf_symbol,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data for {symbol}")

    df = clean_df(df)

    required_cols = ["Date", "Open", "High", "Low", "Close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{symbol} missing column {col}")

    bars = []

    for _, row in df.iterrows():
        try:
            date_val = row["Date"]
            if hasattr(date_val, "to_pydatetime"):
                date_val = date_val.to_pydatetime()

            bars.append({
                "time": date_val.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0) or 0),
            })
        except Exception as e:
            print(f"Skipping bad row for {symbol}: {e}")

    if not bars:
        raise ValueError(f"No valid bars for {symbol}")

    return bars


def main():
    for sym, yf_sym in SYMBOL_MAP.items():
        try:
            bars = fetch(sym, yf_sym)

            path = OUT_DIR / f"{sym}.json"
            path.write_text(json.dumps({"bars": bars}, indent=2))

            print(f"✅ Saved {sym} ({len(bars)} bars)")

        except Exception as e:
            print(f"❌ FAILED {sym}: {e}")


if __name__ == "__main__":
    main()
