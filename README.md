# Indian Stock Pro v1.1

A research-oriented stock ranking PWA with 30 candidates divided into six price bands.

## Price bands
1. ₹20–₹100 — 5 stocks
2. ₹100–₹300 — 5 stocks
3. ₹300–₹500 — 5 stocks
4. ₹500–₹1,000 — 5 stocks
5. ₹1,000–₹1,500 — 5 stocks
6. ₹1,500–₹2,000 — 5 stocks

The ranking engine scores qualifying NSE stocks using momentum, trend, relative strength, volume, RSI quality, MACD, ADX, breakout proximity, volatility and risk/reward.

This is a research tool, not investment advice. It does not guarantee returns.

## Repository layout

- `index.html` — mobile/desktop PWA interface
- `manifest.json` — installable PWA metadata
- `data/rankings.json` — generated ranking output
- `engine/rank_stocks.py` — ranking engine
- `engine/requirements.txt` — Python dependencies
- `.github/workflows/update_rankings.yml` — scheduled/manual ranking update

## GitHub setup

Create a new repository named `IndianStockPro_v1.1`, upload all files, then run:

Actions → Update Stock Rankings → Run workflow.

The workflow also runs automatically on a schedule.

The generated `data/rankings.json` is what the web page reads.
