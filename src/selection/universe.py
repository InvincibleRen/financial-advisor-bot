"""Universe definition and multi-ticker data loading (Direction 1 core, Phase 0).

This module turns the single-ticker data access in ``common/data`` into a
*panel*: prices and fundamentals for every stock in a fixed universe, cached
locally so downstream steps don't repeatedly hit the API.

Design notes
------------
* **Network is isolated.** The public ``load_prices`` / ``load_fundamentals``
  functions never call yfinance directly — they delegate to an injectable
  *fetcher*. The default fetchers use yfinance; tests inject fake fetchers so the
  caching, lagging and ratio logic is verifiable offline (and without a network).
* **Caching.** Each ticker is cached as CSV under ``cache_dir``. On the next run
  the cache is loaded instead of re-fetching, unless ``force_refresh=True``.
* **Look-ahead safety (fundamentals).** yfinance exposes the fiscal *period-end*
  of each quarterly statement but not the public *filing date*. Using the
  period-end as if the numbers were known on that day would leak the future, so
  every fundamental row carries an ``available_date = period_end + reporting_lag``
  (default 60 days) — a conservative approximation of when the figures became
  public. Phase 1 feature construction must join fundamentals on
  ``available_date``, never on ``period_end``.
"""
from __future__ import annotations

import os
import warnings
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.common.timeindex import to_naive_index

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# S&P 500 universe (the "US 500"). Fetched from the Wikipedia constituents list;
# dual-class dots normalised to dashes for yfinance (e.g. BRK.B -> BRK-B). This
# replaces the original bounded ~27-name large-cap set: selection now ranks across
# the whole index. Tickers that fail to fetch are skipped at load time.
DEFAULT_UNIVERSE: List[str] = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES",
    "AFL", "A", "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN",
    "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR",
    "AEE", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "AME",
    "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
    "APP", "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ",
    "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON",
    "BKR", "BALL", "BAC", "BAX", "BDX", "BRK-B", "BBY", "TECH",
    "BIIB", "BLK", "BX", "XYZ", "BNY", "BA", "BKNG", "BSX",
    "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP",
    "CHRW", "CDNS", "CPT", "CPB", "COF", "CAH", "CCL", "CARR",
    "CVNA", "CASY", "CAT", "CBOE", "CBRE", "CDW", "COR", "CNC",
    "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB",
    "CHD", "CIEN", "CI", "CINF", "CTAS", "CSCO", "C", "CFG",
    "CLX", "CME", "CMS", "KO", "CTSH", "COHR", "COIN", "CL",
    "CMCSA", "FIX", "CAG", "COP", "ED", "STZ", "CEG", "COO",
    "CPRT", "GLW", "CPAY", "CTVA", "CSGP", "COST", "CRH", "CRWD",
    "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DDOG", "DVA",
    "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR",
    "DG", "DLTR", "D", "DPZ", "DASH", "DOV", "DOW", "DHI",
    "DTE", "DUK", "DD", "ETN", "EBAY", "SATS", "ECL", "EIX",
    "EW", "EA", "ELV", "EME", "EMR", "ETR", "EOG", "EPAM",
    "EQT", "EFX", "EQIX", "EQR", "ERIE", "ESS", "EL", "EG",
    "EVRG", "ES", "EXC", "EXE", "EXPE", "EXPD", "EXR", "XOM",
    "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB",
    "FSLR", "FE", "FISV", "F", "FTNT", "FTV", "FOXA", "FOX",
    "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN",
    "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL",
    "GDDY", "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC",
    "HSY", "HPE", "HLT", "HD", "HON", "HRL", "HST", "HWM",
    "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX",
    "ITW", "INCY", "IR", "PODD", "INTC", "IBKR", "ICE", "IFF",
    "IP", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT",
    "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "KVUE", "KDP",
    "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC",
    "KR", "LHX", "LH", "LRCX", "LVS", "LDOS", "LEN", "LII",
    "LLY", "LIN", "LYV", "LMT", "L", "LOW", "LULU", "LITE",
    "LYB", "MTB", "MPC", "MAR", "MRSH", "MLM", "MAS", "MA",
    "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD",
    "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "TAP", "MDLZ",
    "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ",
    "NTAP", "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI",
    "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVDA",
    "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE",
    "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PSKY", "PH",
    "PAYX", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX",
    "PNW", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR",
    "PLD", "PRU", "PEG", "PTC", "PSA", "PHM", "PWR", "QCOM",
    "DGX", "Q", "RL", "RJF", "RTX", "O", "REG", "REGN",
    "RF", "RSG", "RMD", "RVTY", "HOOD", "ROK", "ROL", "ROP",
    "ROST", "RCL", "SPGI", "CRM", "SNDK", "SBAC", "SLB", "STX",
    "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA",
    "SOLV", "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE",
    "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO",
    "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA", "TXN",
    "TPL", "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT",
    "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER",
    "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS",
    "VLO", "VEEV", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX",
    "VRT", "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW",
    "WAB", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC",
    "WELL", "WST", "WDC", "WY", "WSM", "WMB", "WTW", "WDAY",
    "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS",
]

# Default assumed delay between a quarter's fiscal period-end and the date its
# figures become public. 60 calendar days is a conservative mid-range estimate
# for US large-caps (10-Q filings are typically due within ~40 days).
DEFAULT_REPORTING_LAG_DAYS = 60

# Fundamental line items the loader standardises on.
_FUNDAMENTAL_RAW_COLUMNS = [
    "net_income", "total_revenue", "total_equity", "total_debt", "shares_outstanding",
]

# Type aliases for the injectable fetchers.
PriceFetcher = Callable[[str, Optional[str], Optional[str]], pd.DataFrame]
FundamentalsFetcher = Callable[[str], pd.DataFrame]


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def _normalise_tickers(tickers: Optional[List[str]]) -> List[str]:
    tickers = list(DEFAULT_UNIVERSE if tickers is None else tickers)
    return [t.upper().strip() for t in tickers]


def _yf_price_fetcher(ticker: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Default price fetcher (yfinance). Not exercised by the offline tests."""
    import yfinance as yf  # local import so the module loads without network deps

    data = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=False)
    if data.empty:
        raise ValueError(f"No price data returned for {ticker}")
    return data.dropna(subset=["Close"])


def load_prices(
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_dir: str = "data/prices",
    force_refresh: bool = False,
    fetcher: Optional[PriceFetcher] = None,
) -> Dict[str, pd.DataFrame]:
    """Return ``{ticker -> daily OHLCV DataFrame}`` for the universe.

    Each ticker is served from ``cache_dir/<TICKER>.csv`` when available; missing
    or ``force_refresh`` tickers are fetched via ``fetcher`` (default: yfinance)
    and written to the cache. Tickers that fail to fetch are skipped with a
    warning rather than aborting the whole run.

    The cache stores each ticker's *full* fetched history, but the returned frame
    is always sliced to the requested ``[start, end]`` window. This is what makes
    ``--start`` actually take effect on repeat runs: without the slice, a cached
    ticker would silently ignore a narrower window and reuse the original range.
    (To *widen* beyond what is cached, re-run with ``force_refresh=True``.)
    """
    tickers = _normalise_tickers(tickers)
    fetcher = fetcher or _yf_price_fetcher
    os.makedirs(cache_dir, exist_ok=True)

    out: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = os.path.join(cache_dir, f"{ticker}.csv")
        if os.path.exists(path) and not force_refresh:
            out[ticker] = _apply_window(_to_total_return_basis(_read_price_cache(path)), start, end)
            continue
        try:
            df = fetcher(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 - keep the batch resilient
            warnings.warn(f"Skipping {ticker}: price fetch failed ({exc})")
            continue
        df = df.dropna(subset=["Close"]) if "Close" in df.columns else df
        _write_price_cache(df, path)          # the cache stays in raw form
        out[ticker] = _apply_window(_to_total_return_basis(df), start, end)
    return out


def _to_total_return_basis(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale OHLC onto a dividend-adjusted (total-return) basis.

    yfinance reports ``Close`` net of splits but *before* dividends, and
    ``Adj Close`` net of both. Momentum, reversal and every realised return in the
    backtest are therefore price returns rather than total returns unless the
    series is adjusted, which systematically penalises high-yield stocks in the
    cross-section (a dividend shows up as a price drop, i.e. as negative momentum).

    Every price column is multiplied by the same per-row factor
    ``Adj Close / Close`` so that the whole OHLC block stays mutually consistent:
    ADX still sees a coherent High/Low/Close, and ``Close`` becomes the
    total-return series. ``Volume`` is a share count and is left untouched (it is
    already split-adjusted, and the factor here carries only the dividend part).
    Frames without an ``Adj Close`` column are returned unchanged.
    """
    if df is None or df.empty or "Adj Close" not in df.columns or "Close" not in df.columns:
        return df

    close = pd.to_numeric(df["Close"], errors="coerce")
    adjusted = pd.to_numeric(df["Adj Close"], errors="coerce")
    factor = adjusted / close.where(close != 0)

    out = df.copy()
    for column in ("Open", "High", "Low", "Close"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") * factor
    return out


def _apply_window(
    df: pd.DataFrame, start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    """Slice a price frame to the requested ``[start, end]`` date window.

    Comparison is done on a timezone-naive copy of the index so it is robust to
    the mixed-offset timestamps a cached yfinance CSV can contain; the original
    frame (and its index) is returned untouched apart from the row selection.
    """
    if df is None or df.empty or (start is None and end is None):
        return df
    naive = to_naive_index(df.index)
    start_ts = pd.Timestamp(start) if start else pd.Timestamp.min
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.max
    mask = (naive >= start_ts) & (naive <= end_ts)
    return df.iloc[mask]


def _write_price_cache(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=True)


def _read_price_cache(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

# Candidate row labels in yfinance statements, mapped to our standard names.
_YF_LABEL_CANDIDATES = {
    "net_income": ["Net Income", "Net Income Common Stockholders", "NetIncome"],
    "total_revenue": ["Total Revenue", "Operating Revenue", "TotalRevenue"],
    "total_equity": [
        "Stockholders Equity", "Total Stockholder Equity",
        "Common Stock Equity", "Total Equity Gross Minority Interest",
    ],
    "total_debt": ["Total Debt", "Net Debt"],
}


def _pick_label(frame: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """Return the first matching row (by label) from a yfinance statement frame."""
    if frame is None or getattr(frame, "empty", True):
        return None
    for label in candidates:
        if label in frame.index:
            return frame.loc[label]
    return None


def _statement_frame(income, balance) -> Optional[pd.DataFrame]:
    """Extract the four raw line items from one (income, balance) statement pair.

    yfinance statements have dates as columns and line items as rows, so this
    transposes and relabels them into a period-end-indexed frame. Returns ``None``
    when the income statement has no usable net-income row.
    """
    net_income = _pick_label(income, _YF_LABEL_CANDIDATES["net_income"])
    if net_income is None:
        return None
    return pd.DataFrame({
        "net_income": net_income,
        "total_revenue": _pick_label(income, _YF_LABEL_CANDIDATES["total_revenue"]),
        "total_equity": _pick_label(balance, _YF_LABEL_CANDIDATES["total_equity"]),
        "total_debt": _pick_label(balance, _YF_LABEL_CANDIDATES["total_debt"]),
    })


def _yf_fundamentals_fetcher(ticker: str) -> pd.DataFrame:
    """Default fundamentals fetcher (yfinance). Not exercised by offline tests.

    Pulls **both** the quarterly and the annual statements and tags each row with a
    ``freq`` column. yfinance only exposes a few recent quarters, so the annual
    statements (roughly four fiscal years) are what extend the usable history
    backwards; the tidy step computes each row's ratios with the right trailing
    window for its frequency and prefers the quarterly row on any overlap.
    """
    import yfinance as yf  # local import

    t = yf.Ticker(ticker)
    frames: List[pd.DataFrame] = []
    for freq, income, balance in (
        ("quarterly", t.quarterly_income_stmt, t.quarterly_balance_sheet),
        ("annual", t.income_stmt, t.balance_sheet),
    ):
        frame = _statement_frame(income, balance)
        if frame is not None and not frame.empty:
            frame = frame.copy()
            frame["freq"] = freq
            frames.append(frame)

    if not frames:
        raise ValueError(f"No income statement available for {ticker}")

    try:
        shares = t.info.get("sharesOutstanding")
    except Exception:  # noqa: BLE001
        shares = None

    combined = pd.concat(frames)
    combined["shares_outstanding"] = shares
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "period_end"
    return combined.sort_index()


def load_fundamentals(
    tickers: Optional[List[str]] = None,
    cache_dir: str = "data/fundamentals",
    reporting_lag_days: int = DEFAULT_REPORTING_LAG_DAYS,
    force_refresh: bool = False,
    fetcher: Optional[FundamentalsFetcher] = None,
) -> pd.DataFrame:
    """Return a tidy, leakage-aware fundamentals table for the universe.

    Columns: ``ticker, period_end, available_date`` plus the raw line items and
    the derived pure-fundamental ratios (``roe, net_margin, debt_to_equity,
    earnings_growth_yoy``). Price-dependent ratios (PE, PB) are deliberately left
    to Phase 1 feature construction, which has prices to hand.

    ``available_date = period_end + reporting_lag_days`` is the earliest date the
    figures may be used as a feature (see module docstring).
    """
    tickers = _normalise_tickers(tickers)
    fetcher = fetcher or _yf_fundamentals_fetcher
    os.makedirs(cache_dir, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        path = os.path.join(cache_dir, f"{ticker}.csv")
        if os.path.exists(path) and not force_refresh:
            frames.append(_read_fundamentals_cache(path))
            continue
        try:
            raw = fetcher(ticker)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skipping {ticker}: fundamentals fetch failed ({exc})")
            continue
        tidy = _tidy_fundamentals(ticker, raw, reporting_lag_days)
        _write_fundamentals_cache(tidy, path)
        frames.append(tidy)

    if not frames:
        return _empty_fundamentals_frame()
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "period_end"]).reset_index(drop=True)


def _ratios_for_freq(sub: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Compute TTM net income and the pure-fundamental ratios for one frequency.

    Quarterly rows use a rolling 4-quarter window for the trailing-twelve-month
    net income and a 4-quarter lag for year-on-year growth. Annual rows already
    cover a full fiscal year, so their TTM is the annual net income itself and the
    year-on-year comparison is the previous annual figure. Handling each frequency
    with its own trailing window is what lets the annual statements extend the
    usable history without corrupting the quarterly TTM logic.
    """
    sub = sub.sort_index().copy()
    if freq == "annual":
        ttm = sub["net_income"]
        prior = sub["net_income"].shift(1)
    else:
        ttm = sub["net_income"].rolling(window=4, min_periods=4).sum()
        prior = sub["net_income"].shift(4)

    sub["ttm_net_income"] = ttm
    sub["roe"] = _safe_div(ttm, sub["total_equity"])
    sub["net_margin"] = _safe_div(sub["net_income"], sub["total_revenue"])
    sub["debt_to_equity"] = _safe_div(sub["total_debt"], sub["total_equity"])
    sub["earnings_growth_yoy"] = _safe_div(sub["net_income"] - prior, prior.abs())
    return sub


def _tidy_fundamentals(ticker: str, raw: pd.DataFrame, reporting_lag_days: int) -> pd.DataFrame:
    """Turn a raw per-ticker statement frame into the tidy, ratio-augmented form.

    The raw frame may carry a ``freq`` column ("quarterly"/"annual"); when absent
    every row is treated as quarterly (backward compatible with the old single
    -frequency fetchers and caches). Ratios are computed independently per
    frequency, then the two are merged and de-duplicated on ``period_end`` with the
    quarterly row winning any overlap (it is the more granular, more recent source).
    """
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    for col in _FUNDAMENTAL_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "freq" not in df.columns:
        df["freq"] = "quarterly"

    parts = [_ratios_for_freq(group, str(freq)) for freq, group in df.groupby("freq")]
    tidy = pd.concat(parts).sort_index()

    tidy = tidy.reset_index().rename(columns={"index": "period_end", "Date": "period_end"})
    if "period_end" not in tidy.columns:
        tidy = tidy.rename(columns={tidy.columns[0]: "period_end"})
    tidy["period_end"] = pd.to_datetime(tidy["period_end"])

    # Prefer the quarterly row when a fiscal-year end coincides with a quarter end.
    tidy["_prefer"] = (tidy["freq"] == "quarterly").astype(int)
    tidy = (
        tidy.sort_values(["period_end", "_prefer"])
        .drop_duplicates("period_end", keep="last")
        .drop(columns=["_prefer", "freq"])
    )

    tidy.insert(0, "ticker", ticker)
    tidy["available_date"] = tidy["period_end"] + pd.Timedelta(days=reporting_lag_days)

    ordered = (
        ["ticker", "period_end", "available_date"]
        + _FUNDAMENTAL_RAW_COLUMNS
        + ["ttm_net_income", "roe", "net_margin", "debt_to_equity", "earnings_growth_yoy"]
    )
    return tidy[ordered].sort_values("period_end").reset_index(drop=True)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN (not inf) on a zero/NA denominator."""
    denom = pd.to_numeric(denominator, errors="coerce")
    denom = denom.where(denom != 0)
    return pd.to_numeric(numerator, errors="coerce") / denom


def _empty_fundamentals_frame() -> pd.DataFrame:
    cols = (
        ["ticker", "period_end", "available_date"]
        + _FUNDAMENTAL_RAW_COLUMNS
        + ["ttm_net_income", "roe", "net_margin", "debt_to_equity", "earnings_growth_yoy"]
    )
    return pd.DataFrame(columns=cols)


def _write_fundamentals_cache(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def _read_fundamentals_cache(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["period_end", "available_date"])


# ---------------------------------------------------------------------------
# Sectors (for sector-neutral feature normalisation)
# ---------------------------------------------------------------------------

SectorFetcher = Callable[[str], Optional[str]]


def _yf_sector_fetcher(ticker: str) -> Optional[str]:
    """Default sector fetcher (yfinance). Not exercised by the offline tests."""
    import yfinance as yf  # local import

    try:
        return yf.Ticker(ticker).info.get("sector")
    except Exception:  # noqa: BLE001 - keep the batch resilient
        return None


def load_sectors(
    tickers: Optional[List[str]] = None,
    cache_path: str = "data/sectors.csv",
    force_refresh: bool = False,
    fetcher: Optional[SectorFetcher] = None,
) -> pd.Series:
    """Return a ``ticker -> sector`` Series, cached to a single CSV.

    Sector is static, so the whole universe is cached in one small file; only
    tickers missing from the cache are fetched (via ``fetcher``, default yfinance).
    Consumed by :func:`selection.features.cross_sectional_normalize` for
    sector-neutral ranking. Tickers whose sector cannot be resolved are simply
    omitted (the normaliser then pools them into a shared bucket).
    """
    tickers = _normalise_tickers(tickers)
    fetcher = fetcher or _yf_sector_fetcher

    cached: Dict[str, str] = {}
    if os.path.exists(cache_path) and not force_refresh:
        prior = pd.read_csv(cache_path)
        cached = {str(r.ticker): str(r.sector) for r in prior.itertuples()
                  if isinstance(r.sector, str) and r.sector and r.sector != "nan"}

    resolved: Dict[str, str] = {}
    for ticker in tickers:
        if ticker in cached:
            resolved[ticker] = cached[ticker]
            continue
        sector = fetcher(ticker)
        if sector:
            resolved[ticker] = str(sector)

    if resolved:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        merged = {**cached, **resolved}
        pd.DataFrame({"ticker": list(merged), "sector": list(merged.values())}).to_csv(
            cache_path, index=False
        )
    return pd.Series(resolved, name="sector")
