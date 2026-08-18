"""
Stationarity testing / differencing helpers for the classical models.

Two levels of API:
    check_stationarity(s)   -> full dict of ADF + KPSS numbers and a verdict
    is_stationary(s)        -> plain True/False

and the "what do I do about it" side, which is what ARIMA needs:
    find_d(s)               -> order of differencing that makes s stationary
    make_stationary(s)      -> the differenced series + how it got there
    stationarity_report(df) -> run the above over every row of a DataFrame
"""

import warnings

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf

__all__ = [
    "check_stationarity",
    "is_stationary",
    "find_d",
    "make_stationary",
    "stationarity_report",
]

# adfuller needs enough points for its lag search; below this it fails with a
# confusing internal error, so we check up front and say something useful.
MIN_OBS = 12


def _clean(series: pd.Series) -> pd.Series:
    """Series -> float Series with NaNs/infs dropped."""
    s = pd.Series(series).astype(float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def check_stationarity(series: pd.Series, alpha=0.05, regression="c"):
    """
    ADF  -> H0: unit root (non-stationary).  p < alpha  => stationary
    KPSS -> H0: stationary.                  p < alpha  => non-stationary
    regression: "c" (level stationary) or "ct" (trend stationary)
    """
    s = _clean(series)

    adf_stat, adf_p, adf_lags, adf_n, adf_crit, _ = adfuller(
        s, autolag="AIC", regression=regression
    )

    # KPSS warns when p-value is outside the lookup table; that's fine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
            s, regression=regression, nlags="auto"
        )

    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p > alpha

    if adf_stationary and kpss_stationary:
        verdict = "stationary"
    elif not adf_stationary and not kpss_stationary:
        verdict = "non-stationary (try differencing)"
    elif adf_stationary and not kpss_stationary:
        verdict = "difference-stationary conflict: likely trend-stationary -> detrend"
    else:
        verdict = "conflict: likely difference-stationary -> difference the series"

    return {
        "n": len(s),
        "adf_stat": adf_stat, "adf_p": adf_p, "adf_lags": adf_lags,
        "adf_crit": adf_crit, "adf_says_stationary": adf_stationary,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p, "kpss_lags": kpss_lags,
        "kpss_crit": kpss_crit, "kpss_says_stationary": kpss_stationary,
        "verdict": verdict,
    }


def is_stationary(series: pd.Series, alpha=0.05, regression="c", strict=True) -> bool:
    """
    True  -> the series can be treated as stationary
    False -> it needs a transform (differencing / detrending) first

    strict=True  : both ADF and KPSS must agree it is stationary.
    strict=False : ADF alone decides (KPSS is ignored).

    Raises ValueError if there is not enough usable data to run the tests.
    """
    s = _clean(series)

    if len(s) < MIN_OBS:
        raise ValueError(f"need at least {MIN_OBS} non-NaN observations, got {len(s)}")

    # A flat series has zero variance -> the tests are undefined, but a
    # constant is trivially stationary.
    if np.isclose(s.std(ddof=0), 0.0):
        return True

    adf_p = adfuller(s, autolag="AIC", regression=regression)[1]
    adf_stationary = adf_p < alpha

    if not strict:
        return bool(adf_stationary)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_p = kpss(s, regression=regression, nlags="auto")[1]
    kpss_stationary = kpss_p > alpha

    return bool(adf_stationary and kpss_stationary)


def find_d(series: pd.Series, max_d=2, alpha=0.05, regression="c", strict=True) -> int:
    """
    Smallest number of regular differences that makes the series stationary --
    the "d" of ARIMA(p, d, q).

    Returns max_d if it never passes within the budget; over-differencing is
    the usual reason to keep max_d at 2, so a returned max_d is a hint to look
    at the series by hand rather than a confident answer.

    Raises ValueError if the series is too short to test at all.
    """
    s = _clean(series)

    for d in range(max_d + 1):
        # Each difference costs an observation. Too short to even start is a
        # real error; running out part way just ends the search early.
        if len(s) < MIN_OBS:
            if d == 0:
                raise ValueError(
                    f"need at least {MIN_OBS} non-NaN observations, got {len(s)}"
                )
            break
        if is_stationary(s, alpha=alpha, regression=regression, strict=strict):
            return d
        s = _diff(s, 1)

    return max_d


def _diff(series: pd.Series, d: int, lag=1) -> pd.Series:
    """Apply .diff(lag) d times -- NOT the same as .diff(d)."""
    s = series
    for _ in range(d):
        s = s.diff(lag).dropna()
    return s


def _seasonal_strength(series: pd.Series, period: int) -> float:
    """
    How much the seasonal lag stands out in the autocorrelation, 0 if not at all.

    Measured on the first difference: a trending or random-walk series has high
    autocorrelation at *every* lag, so raw ACF at the seasonal lag would call
    any trend seasonal. Differencing strips the trend and leaves the cycle.
    The lag must also beat its neighbours, otherwise what we are seeing is
    leftover decay rather than a peak at the season.
    """
    s = _diff(_clean(series), 1)
    if len(s) <= period + 2 or np.isclose(s.std(ddof=0), 0.0):
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = acf(s, nlags=period + 1, fft=True)

    if a[period] <= max(a[period - 1], a[period + 1]):
        return 0.0
    return float(a[period])


def make_stationary(
    series: pd.Series,
    max_d=2,
    seasonal_period=None,
    max_D=1,
    seasonal_threshold=0.3,
    alpha=0.05,
    regression="c",
    strict=True,
):
    """
    Difference a series until it is stationary and report what was done.

    Seasonal differencing (lag = seasonal_period, e.g. 7 for daily retail data)
    is applied first, and only while the autocorrelation at that lag is above
    seasonal_threshold -- the usual auto-ARIMA style heuristic. Regular
    differencing then handles whatever trend is left.

    Returns (transformed_series, info) where info holds:
        d, D, seasonal_period, stationary (did it actually get there), n
    """
    s = _clean(series)
    D = 0

    if seasonal_period:
        while (
            D < max_D
            and len(s) > seasonal_period + MIN_OBS
            and _seasonal_strength(s, seasonal_period) > seasonal_threshold
        ):
            s = s.diff(seasonal_period).dropna()
            D += 1

    d = find_d(s, max_d=max_d, alpha=alpha, regression=regression, strict=strict)
    s = _diff(s, d)

    # find_d returns max_d even when it never passed, so confirm the result.
    try:
        stationary = is_stationary(s, alpha=alpha, regression=regression, strict=strict)
    except ValueError:
        stationary = False

    info = {
        "d": d,
        "D": D,
        "seasonal_period": seasonal_period if D else None,
        "stationary": stationary,
        "n": len(s),
    }
    return s, info


def stationarity_report(data, seasonal_period=None, **kwargs) -> pd.DataFrame:
    """
    Run make_stationary over many series at once.

    Accepts a DataFrame laid out the way Data.get_train_data returns it -- one
    row per product, one column per day -- or a single Series. Returns one row
    per input series: the index label, length, d, D, and whether it ended up
    stationary. Series too short to test are reported with stationary = False
    and d = D = NaN instead of blowing up the whole run.
    """
    if isinstance(data, pd.Series):
        rows = [(data.name, data)]
    else:
        rows = list(pd.DataFrame(data).iterrows())

    out = []
    for label, row in rows:
        series = pd.Series(row)
        try:
            _, info = make_stationary(series, seasonal_period=seasonal_period, **kwargs)
            info["error"] = None
        except ValueError as e:
            info = {
                "d": np.nan, "D": np.nan, "seasonal_period": None,
                "stationary": False, "n": len(_clean(series)), "error": str(e),
            }
        out.append({"id": label, **info})

    return pd.DataFrame(out)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    demo = {
        "white noise": pd.Series(rng.normal(size=400)),
        "random walk": pd.Series(rng.normal(size=400).cumsum()),
        "trend + noise": pd.Series(np.arange(400) * 0.1 + rng.normal(size=400)),
        "weekly seasonal": pd.Series(
            10 + 5 * np.sin(np.arange(400) * 2 * np.pi / 7) + rng.normal(size=400)
        ),
    }

    for name, s in demo.items():
        res = check_stationarity(s)
        _, info = make_stationary(s, seasonal_period=7)
        print(f"{name:>16} | ADF p={res['adf_p']:.4f} KPSS p={res['kpss_p']:.4f} "
              f"| d={info['d']} D={info['D']} -> stationary={info['stationary']}")

    print()
    print(stationarity_report(pd.DataFrame(demo).T, seasonal_period=7))
