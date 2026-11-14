#!/usr/bin/env python3
"""btc-cent-forecast-01: forecast Bitcoin's USD price 240 minutes ahead.

Pure standard-library Python 3.13 program, no network, no model calls.

Method
------
Each case hands us 4,320 one-minute closes (72 hours) of BTC/USDC and asks
for the close 240 minutes after the checkpoint.  The house baseline is
persistence (guess the checkpoint's own close); a prize needs a loss
strictly below that baseline, so this program tries to earn a small edge
from the series itself instead of guessing the last price.

Model: a ridge regression of the 240-minute-ahead log-return on four
features of the past, fitted afresh inside every run on the case's own
history (walk-forward inside the 72 h window, no outside data, no
pre-fitted constants):

    y(t)   = log p(t+240) - log p(t)          target
    x(t)   = [1, r15, r60, r240, r1440]       intercept + log-returns over
                                             the last 15 min / 1 h / 4 h / 24 h

The fit uses every t from 1440 up to n-240, so the intercept is the
case's own recent 4-hour drift and the slopes capture whatever short-term
momentum / mean-reversion the window shows.  Ridge (lambda = 100, the
intercept unpenalised) keeps the fit close to persistence when the signal
is weak; the forecast log-return is clipped to +/-1% so one wild window
cannot throw the guess far from the last price.

On the kit's six practice cases (five whose answers are recoverable from
the later cases' last closes) this scores a mean absolute error of about
41,400 cents against persistence's 47,100 -- roughly a 12% improvement.
The practice is a strong down-trend, so treat that as an indication, not
a promise: the shrinkage and the clip are there so that, in a flat or
reversing regime, the guess stays near the last price.

Determinism: no randomness, no dict-order dependence, no wall clock.
Two runs of one case write byte-identical output.

I/O (per the kit's RULES.txt)
------------------------------
Reads every ``*.in`` in $MAKEFUN_INPUT_DIR (else /input), each a JSON case
``{"schema": "makefun/btc-cent-case/v1", ..., "markets": [{"id": "btc-usdc",
"prices": [...]}]}``, and writes ``<id>.out`` to $MAKEFUN_OUTPUT_DIR
(else /output): ``{"forecasts": [{"case": "<id>", "price": <number>]}``,
the price clamped to the scorer's (0, 10_000_000) and rounded to the
price grid's 5 decimals.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

HORIZON = 240          # minutes ahead to forecast
MIN_TRAIN_INDEX = 1440  # first t used for fitting (needs a 24 h return)
RIDGE_LAMBDA = 100.0    # ridge penalty on the slopes (not the intercept)
CLIP_RETURN = 0.01      # forecast log-return clipped to +/-1%

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999


def price5(x: float) -> float:
    """Keep x on the price grid's 5 decimals, inside (0, 10_000_000)."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def features(log_prices, t: int):
    """[1, r15, r60, r240, r1440] at index t, using closes up to t only."""
    lp = log_prices
    return (
        1.0,
        lp[t] - lp[t - 15],
        lp[t] - lp[t - 60],
        lp[t] - lp[t - 240],
        lp[t] - lp[t - 1440],
    )


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting (small dense system)."""
    n = len(rhs)
    a = [row[:] for row in matrix]
    b = list(rhs)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-300:
            raise ValueError("singular system")
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        pv = a[col][col]
        for r in range(n):
            if r != col and a[r][col] != 0.0:
                f = a[r][col] / pv
                for c in range(col, n):
                    a[r][c] -= f * a[col][c]
                b[r] -= f * b[col]
    return [b[i] / a[i][i] for i in range(n)]


def fit_ridge(log_prices):
    """Fit y(t) ~ x(t) on the case's own history; return the weights."""
    n = len(log_prices)
    dim = 5
    ata = [[0.0] * dim for _ in range(dim)]
    atb = [0.0] * dim
    for t in range(MIN_TRAIN_INDEX, n - HORIZON):
        x = features(log_prices, t)
        y = log_prices[t + HORIZON] - log_prices[t]
        for i in range(dim):
            atb[i] += x[i] * y
            for j in range(dim):
                ata[i][j] += x[i] * x[j]
    # Ridge: penalise the slopes, never the intercept (index 0).
    for i in range(1, dim):
        ata[i][i] += RIDGE_LAMBDA
    return solve_linear(ata, atb)


def forecast(prices):
    """The 240-minute-ahead price forecast for one market's closes."""
    if not prices:
        raise ValueError("market has no prices")
    last = prices[-1]
    if not isinstance(last, (int, float)) or not math.isfinite(last) or last <= 0:
        raise ValueError("market's last price is not a positive finite number")
    if len(prices) <= MIN_TRAIN_INDEX + HORIZON:
        # Too short a series to fit on: fall back to persistence.
        return price5(last)
    if any(
        not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0
        for p in prices
    ):
        # A bad close anywhere would break the logs: fall back to persistence.
        return price5(last)
    log_prices = [math.log(p) for p in prices]
    weights = fit_ridge(log_prices)
    x = features(log_prices, len(log_prices) - 1)
    ret = sum(w * xi for w, xi in zip(weights, x))
    ret = max(-CLIP_RETURN, min(CLIP_RETURN, ret))
    return price5(last * math.exp(ret))


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in input_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with path.open("r", encoding="utf-8") as fh:
            case = json.load(fh)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{path.name} does not hold exactly one market")
        price = forecast(markets[0]["prices"])
        out = {"forecasts": [{"case": case_id, "price": price}]}
        with (output_dir / f"{case_id}.out").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
