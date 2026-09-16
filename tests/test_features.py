"""Offline tests for the Phase 1 leakage-safe feature matrix.

Deterministic synthetic data; no network. The tests assert the property that
matters most for a backtest's credibility: features at ``t`` depend only on
information available at ``t`` (technical as-of close, fundamentals as-of their
``available_date``), plus the correctness of the price-dependent valuation ratios.
"""
import numpy as np
import pandas as pd

from src.selection import features as F


# --------------------------------------------------------------------------- #
# Builders                                                                     #
# --------------------------------------------------------------------------- #

def _prices(close=120.0, start="2019-06-01", periods=500):
    idx = pd.date_range(start, periods=periods, freq="D")
    return {"A": pd.DataFrame({"Close": [close] * periods}, index=idx)}


def _fundamentals(rows):
    """Assemble a tidy fundamentals frame in the Phase 0 column layout."""
    cols = [
        "ticker", "period_end", "available_date", "net_income", "total_revenue",
        "total_equity", "total_debt", "shares_outstanding", "roe", "net_margin",
        "debt_to_equity", "earnings_growth_yoy",
    ]
    df = pd.DataFrame(rows)[cols]
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["available_date"] = pd.to_datetime(df["available_date"])
    return df


def _quarter(ticker, period_end, available_date, net_income, **overrides):
    row = {
        "ticker": ticker, "period_end": period_end, "available_date": available_date,
        "net_income": net_income, "total_revenue": 100.0, "total_equity": 200.0,
        "total_debt": 100.0, "shares_outstanding": 1000.0, "roe": 0.20,
        "net_margin": 0.10, "debt_to_equity": 0.50, "earnings_growth_yoy": 0.0,
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------- #
# Structure                                                                    #
# --------------------------------------------------------------------------- #

def test_matrix_shape_and_index():
    prices = _prices()
    funds = _fundamentals([_quarter("A", "2019-03-31", "2019-05-30", 10.0)])
    t = pd.Timestamp("2020-01-15")

    matrix = F.build_feature_matrix(prices, funds, [t])

    assert list(matrix.columns) == F.TECHNICAL_FEATURES + F.FUNDAMENTAL_FEATURES
    assert list(matrix.index.names) == ["rebalance_date", "ticker"]
    assert (t, "A") in matrix.index


# --------------------------------------------------------------------------- #
# Valuation ratios                                                             #
# --------------------------------------------------------------------------- #

def test_pe_and_pb_use_price_and_ttm():
    # Four quarters of net income 10 -> TTM = 40. Price 120, shares 1000 ->
    # market cap 120_000. PE = 120_000 / 40 = 3000; PB = 120_000 / 200 = 600.
    prices = _prices(close=120.0)
    funds = _fundamentals([
        _quarter("A", "2019-03-31", "2019-05-30", 10.0),
        _quarter("A", "2019-06-30", "2019-08-29", 10.0),
        _quarter("A", "2019-09-30", "2019-11-29", 10.0),
        _quarter("A", "2019-12-31", "2020-02-29", 10.0),
    ])
    t = pd.Timestamp("2020-03-15")  # all four filings public by now

    row = F.build_feature_matrix(prices, funds, [t]).loc[(t, "A")]

    assert np.isclose(row["pe"], 3000.0)
    assert np.isclose(row["pb"], 600.0)
    assert np.isclose(row["roe"], 0.20)
    assert np.isclose(row["net_margin"], 0.10)
    assert np.isclose(row["debt_to_equity"], 0.50)


# --------------------------------------------------------------------------- #
# As-of fundamentals (the core leakage guard)                                 #
# --------------------------------------------------------------------------- #

def test_fundamentals_join_uses_available_date():
    prices = _prices(periods=520)
    funds = _fundamentals([
        _quarter("A", "2019-12-31", "2020-03-01", 10.0, roe=0.10),
        _quarter("A", "2020-06-30", "2020-09-01", 10.0, roe=0.20),
    ])

    before = pd.Timestamp("2020-05-01")   # only the first filing is public
    after = pd.Timestamp("2020-10-01")    # both are public -> most recent wins

    m_before = F.build_feature_matrix(prices, funds, [before])
    m_after = F.build_feature_matrix(prices, funds, [after])

    assert np.isclose(m_before.loc[(before, "A"), "roe"], 0.10)
    assert np.isclose(m_after.loc[(after, "A"), "roe"], 0.20)


def test_future_filing_is_not_used_before_available_date():
    prices = _prices()  # price history starts 2019-06-01
    funds = _fundamentals([_quarter("A", "2019-06-30", "2019-08-30", 10.0, roe=0.10)])
    t = pd.Timestamp("2019-07-01")  # in price history, before the filing's available_date

    row = F.build_feature_matrix(prices, funds, [t]).loc[(t, "A")]
    assert np.isnan(row["roe"])  # nothing public yet -> NaN, not a leaked value


# --------------------------------------------------------------------------- #
# Leakage invariant: the future cannot change the past                        #
# --------------------------------------------------------------------------- #

def test_future_data_does_not_change_features_at_t():
    prices = _prices(close=120.0, periods=400)
    funds = _fundamentals([
        _quarter("A", "2019-03-31", "2019-05-30", 10.0),
        _quarter("A", "2019-06-30", "2019-08-29", 10.0),
        _quarter("A", "2019-09-30", "2019-11-29", 10.0),
        _quarter("A", "2019-12-31", "2020-02-29", 10.0),
    ])
    t = pd.Timestamp("2020-03-15")
    baseline = F.build_feature_matrix(prices, funds, [t]).loc[(t, "A")]

    # Inject a wild FUTURE price spike and a FUTURE filing (available after t).
    spiked = {"A": prices["A"].copy()}
    future_idx = pd.date_range("2020-05-01", periods=30, freq="D")
    spiked["A"] = pd.concat([
        spiked["A"],
        pd.DataFrame({"Close": [10_000.0] * 30}, index=future_idx),
    ])
    funds_future = pd.concat([
        funds,
        _fundamentals([_quarter("A", "2020-03-31", "2020-05-30", 9_999.0, roe=9.9)]),
    ], ignore_index=True)

    perturbed = F.build_feature_matrix(spiked, funds_future, [t]).loc[(t, "A")]
    pd.testing.assert_series_equal(baseline, perturbed)


# --------------------------------------------------------------------------- #
# Robustness                                                                   #
# --------------------------------------------------------------------------- #

def test_missing_fundamentals_yield_nan_but_keep_technicals():
    prices = _prices()
    empty = F._augmented_fundamentals_by_ticker(pd.DataFrame())  # sanity: no crash
    assert empty == {}

    matrix = F.build_feature_matrix(prices, pd.DataFrame(), [pd.Timestamp("2020-01-15")])
    row = matrix.loc[(pd.Timestamp("2020-01-15"), "A")]
    assert row[F.FUNDAMENTAL_FEATURES].isna().all()
    assert not np.isnan(row["mom_1m"])  # technicals still present


def test_optional_sentiment_is_left_joined():
    prices = _prices()
    funds = _fundamentals([_quarter("A", "2019-03-31", "2019-05-30", 10.0)])
    t = pd.Timestamp("2020-01-15")
    sentiment = pd.DataFrame({"rebalance_date": [t], "ticker": ["A"], "sentiment": [0.42]})

    matrix = F.build_feature_matrix(prices, funds, [t], sentiment=sentiment)
    assert "sentiment" in matrix.columns
    assert np.isclose(matrix.loc[(t, "A"), "sentiment"], 0.42)


# --------------------------------------------------------------------------- #
# Cross-sectional normalisation (accuracy lever)                              #
# --------------------------------------------------------------------------- #

def _toy_matrix():
    """Two dates x three tickers, one feature column with known ordering."""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2020-01-31"), t) for t in ("A", "B", "C")]
        + [(pd.Timestamp("2020-02-29"), t) for t in ("A", "B", "C")],
        names=["rebalance_date", "ticker"],
    )
    return pd.DataFrame({"mom_1m": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]}, index=idx)


def test_rank_normalize_is_within_date_and_centred():
    out = F.cross_sectional_normalize(_toy_matrix(), method="rank")
    jan = out.xs(pd.Timestamp("2020-01-31"), level="rebalance_date")["mom_1m"]
    feb = out.xs(pd.Timestamp("2020-02-29"), level="rebalance_date")["mom_1m"]
    # Ranks depend only on within-date ordering, not absolute levels, so the two
    # dates (1,2,3 vs 10,20,30) map to the *same* percentiles.
    assert list(jan.round(6)) == list(feb.round(6))
    # Monotonically increasing with the raw values, lowest below 0, and bounded
    # within [-0.5, 0.5].
    assert jan.iloc[0] < jan.iloc[1] < jan.iloc[2]
    assert jan.iloc[0] < 0
    assert jan.min() >= -0.5 and jan.max() <= 0.5


def test_normalize_is_leakage_safe_across_dates():
    """A date's normalised values depend only on that date's own cross-section.

    Removing an unrelated future date must leave an earlier date's values
    unchanged — the same as-of invariant the raw features satisfy.
    """
    full = F.cross_sectional_normalize(_toy_matrix(), method="rank")
    jan_only_input = _toy_matrix().xs(
        pd.Timestamp("2020-01-31"), level="rebalance_date", drop_level=False
    )
    jan_only = F.cross_sectional_normalize(jan_only_input, method="rank")
    pd.testing.assert_frame_equal(
        full.xs(pd.Timestamp("2020-01-31"), level="rebalance_date", drop_level=False),
        jan_only,
    )


def test_normalize_preserves_nans():
    m = _toy_matrix()
    m.iloc[0, 0] = np.nan
    out = F.cross_sectional_normalize(m, method="rank")
    assert np.isnan(out.iloc[0, 0])


def test_sector_neutral_ranks_within_sector():
    """With a sector map, ranking is within (date, sector), not the whole date."""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2020-01-31"), t) for t in ("A", "B", "C", "D")],
        names=["rebalance_date", "ticker"],
    )
    # A,B in sector X; C,D in sector Y. Raw values ascending A<B<C<D.
    m = pd.DataFrame({"mom_1m": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    sectors = pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Y"})

    out = F.cross_sectional_normalize(m, method="rank", sectors=sectors)["mom_1m"]
    # Within each 2-stock sector the lower name gets the bottom percentile and the
    # higher the top, so the two sectors produce the *same* pair of ranks; the
    # cross-sector level (C,D > A,B) is neutralised away.
    assert out.loc[(pd.Timestamp("2020-01-31"), "A")] == out.loc[(pd.Timestamp("2020-01-31"), "C")]
    assert out.loc[(pd.Timestamp("2020-01-31"), "B")] == out.loc[(pd.Timestamp("2020-01-31"), "D")]
    assert out.loc[(pd.Timestamp("2020-01-31"), "A")] < out.loc[(pd.Timestamp("2020-01-31"), "B")]
