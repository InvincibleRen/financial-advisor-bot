"""Command-line demo for the FinBERT news sentiment feature (Extension A).

Fetches recent headlines for each ticker and prints a FinBERT sentiment score in
[-1, 1] (positive - negative), plus a couple of example headlines. This is a
live "sentiment right now" reading — yfinance only exposes recent news, so it is
a demonstration of the Extension-A integration rather than a historical backtest.

Requires the Extension-A dependencies:

    pip install -r requirements-extension-a.txt

Examples
--------
    python -m src.cli.sentiment_cli AAPL MSFT NVDA
    python -m src.cli.sentiment_cli AAPL --lookback-days 14
"""
from __future__ import annotations

import argparse
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.selection import universe as U
from src.sentiment.finbert import FinBERTScorer
from src.sentiment.news_source import YFinanceNewsSource


def _label(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="FinBERT news-sentiment demo (Extension A)")
    parser.add_argument("tickers", nargs="*", default=None,
                        help="Tickers to score (default: the built-in DEFAULT_UNIVERSE)")
    parser.add_argument("--asof", default=None, help="As-of date (YYYY-MM-DD); default: today")
    parser.add_argument("--lookback-days", type=int, default=7,
                        help="Headline window in days ending at --asof (default 7)")
    parser.add_argument("--backend", choices=["torch", "onnx", "onnx-local"], default="torch",
                        help="Inference backend (default: torch). On x86 macOS / Intel Mac "
                             "use 'onnx-local' — a torch-free ONNX Runtime path.")
    parser.add_argument("--model", default="ProsusAI/finbert",
                        help="Hugging Face model id (default: ProsusAI/finbert)")
    args = parser.parse_args(argv)

    tickers = [t.upper().strip() for t in args.tickers] if args.tickers else list(U.DEFAULT_UNIVERSE)
    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(datetime.now().date())
    source = YFinanceNewsSource()
    scorer = FinBERTScorer(model_name=args.model, backend=args.backend)  # loaded lazily

    print(f"News sentiment as of {asof.date()} (lookback {args.lookback_days}d) — {len(tickers)} tickers")
    print("=" * 60)
    scores = []
    for ticker in tickers:
        headlines = source.fetch_headlines(ticker, asof, args.lookback_days)
        if not headlines:
            print(f"{ticker:>6}:  no recent headlines")
            continue
        score = scorer.score_texts([h.text for h in headlines])
        scores.append(score)
        print(f"{ticker:>6}:  {score:+.3f}  ({_label(score)})  from {len(headlines)} headlines")
        for h in headlines[:2]:
            print(f"         · {h.text[:80]}")
    print("=" * 60)
    if scores:
        avg = sum(scores) / len(scores)
        print(f"Market mood: average {avg:+.3f} ({_label(avg)}) across {len(scores)} tickers with news.")
    else:
        print("No headlines found for any ticker in this window — try a longer --lookback-days.")
    print("Score in [-1, 1] = mean(P(positive) - P(negative)) across headlines.")


if __name__ == "__main__":
    main()
