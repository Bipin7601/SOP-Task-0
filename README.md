# Swing trading system (NSE, Zerodha Kite)

A rules-based momentum breakout swing strategy for NSE stocks, held for days to a few weeks.
Python code decides the signals, so the same data always gives the same answer. Claude runs the
daily routine through the Kite connector, and **every order needs a human yes**.

## Status

| Phase | State |
|---|---|
| 1. Strategy code, simulator, tests | ✅ Built |
| 2. Download history (Kite Connect data API) | ⏳ Waiting on API key |
| 3. Backtest, in-sample vs out-of-sample, go/no-go | ⏳ Needs data |
| 4. Paper trading, 4–6 weeks | — |
| 5. Live with ₹15,000, approval on every trade | — |

## The strategy

1. **Market filter:** new trades only while Nifty 50 closes above its 200-day average.
2. **Universe:** liquid Nifty 500 stocks (₹5 cr+ average daily traded value, price ≥ ₹50).
3. **Trend template:** close > SMA50 > SMA150 > SMA200, SMA200 rising, close within 25% of its
   52-week high and at least 25% above its 52-week low, relative strength in the top 30%.
4. **Setup:** a tight 15-day base (≤ 12% deep, volatility contracting).
5. **Entry:** close above the base on at least 1.5× average volume; buy at the next open
   (skipped if it gaps more than 3% higher).
6. **Exits:**
   - stop 0.5% below the base low, 3–8% from entry (wider setups are skipped)
   - target at 2.5R
   - stop moves to breakeven at +1R, then trails at 3 × ATR below the highest close
   - time exit after 20 trading days

Every number is in [`config/strategy.yaml`](config/strategy.yaml).

## Risk rules

- 1.5% of equity at risk per trade; at most 3 positions; no position over 40% of equity.
- If equity falls 10% below its peak, no new trades until a human review.

## Backtest realism

- Zerodha delivery charges (STT, exchange, SEBI, stamp duty, GST, ₹15.34 DP per sell) plus 0.1% slippage each side.
- Signals use only data up to that day's close; trades happen the next day.
- Gaps through a stop fill at the open; when a bar touches both stop and target, the stop counts.
- Whole shares only, with the real ₹15,000 starting capital.
- Tuned on 2012–2020, judged on 2021 onward. Includes a Monte Carlo drawdown check and a Nifty 50 comparison.
- Known bias: the universe is today's Nifty 500, so delisted stocks are missing and results look somewhat better than reality.

## Usage

```bash
pip install -r requirements.txt
python -m pytest                                 # tests (synthetic data)

# Needs KITE_API_KEY / KITE_API_SECRET and data/universe/nifty500.csv from NSE
python -m swing.kite_data login                  # once per trading day
python -m swing.kite_data download               # first run ~10 years; afterwards only new bars

python -m swing.cli backtest                     # writes reports/<timestamp>/report.md
python -m swing.cli scan --equity 15000          # today's candidates with entry, stop, target and quantity
```

## Layout

```
config/strategy.yaml   every rule and number
swing/indicators.py    moving averages, ATR, relative strength
swing/signals.py       market filter, trend template, base, breakout
swing/backtest.py      day-by-day portfolio simulator
swing/costs.py         Zerodha delivery charges
swing/metrics.py       stats, Monte Carlo, go/no-go check
swing/kite_data.py     Kite Connect historical download
swing/cli.py           backtest and scan commands
tests/                 unit and end-to-end tests
```

No profits are promised. This is a process for testing a strategy honestly before risking money.
