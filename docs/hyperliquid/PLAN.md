# TradingAgents × Hyperliquid — research and implementation plan

Status: written **before** any code or any simulation was run (2026-10-31), so the
evaluation protocol below is pre-registered. Anything decided after seeing results
is flagged as such in the papers.

## 0. Goal

1. Make TradingAgents trade **BTC perpetual futures on Hyperliquid (HL)**, with the venue's
   real costs: trading fees, hourly funding, slippage, margin and liquidation.
2. Run a **walk-forward simulation over October 2026**, where every decision is made with
   only the information available at its timestamp. No step may see step *n+1*.
3. Write it up as two papers (`papers/`): a methods paper (microstructure, cost model,
   accounting, leakage-free protocol, statistics) and an empirical paper (the October run).

## 1. What upstream already gives us (v0.7.1) and what it does not

| Upstream piece | Reuse? | Gap for HL perps |
|---|---|---|
| LangGraph pipeline: 4 analysts → bull/bear debate → research manager → trader → 3 risk debaters → portfolio manager (PM) | yes, unchanged | none |
| `asset_type="crypto"` (drops the fundamentals analyst) | yes | nothing replaces fundamentals; a **derivatives analyst** fills the slot |
| Point-in-time guards (`as_of`, `as_of_window`, `in_window`) at **date** granularity | yes | 24/7 market: a bar is admissible only if it **closed** before the cutoff instant |
| Price data: Yahoo `BTC-USD` | no | venue prices differ (perp vs spot index); use HL candles |
| Sentiment: StockTwits + Reddit + Yahoo news | no, for history | live-only feeds whose upvotes and selection are measured *today*, which leaks; use archived sources |
| News: Yahoo (recent only), Alpha Vantage (key) | no | add keyless point-in-time vendors: GDELT (`seendate`), Google News RSS (date-only stamps, so lagged) |
| Memory log and reflection, settled with Yahoo closes vs SPY | partly | wrong instrument, benchmark and timing; settle from the simulator's own net P&L instead |
| `backtest.py`: scores ratings, deliberately not a portfolio simulator | no | needs a perp account: fees, funding, margin, position carried day to day |
| Ratings: Buy/Overweight/Hold/Underweight/Sell | yes | need a deterministic map from rating to **signed target exposure** (shorting allowed) |

## 2. Hyperliquid facts the model relies on (checked against the HL docs on 2026-10-31)

- **Fees (perps, tier 0, 14-day volume < $5M):** taker **4.5 bp**, maker **1.5 bp**. Tiers go
  down to 2.4 bp / 0 bp. HYPE staking discounts range from 5% to 40%.
- **Funding:** paid **hourly**, at 1/8 of an 8-hour rate,
  `F_8h = P + clamp(I − P, −0.05%, +0.05%)`, with `I = 0.01%/8h` (0.00125%/h).
  - `P` is the hourly average of the premium index, sampled every 5 s from impact bid/ask
    against the oracle.
  - The payment is `size × oracle_px × rate`; longs pay when it is positive.
  - Cap: 4%/h.
  - Consequence: whenever |I − P| ≤ 5 bp, funding is **exactly** 0.00125%/h. A passive long
    then pays about 10.95%/yr (simple) — a carry cost that matters for buy-and-hold.
- **Margin:** BTC max leverage is 40× (tiered to 20× above $150M notional). Initial margin is
  `notional / leverage`; maintenance is half the initial margin at max leverage, i.e.
  **1.25% of notional**.
- **Market data (public `POST /info`):**
  - `candleSnapshot` keeps the most recent 5000 candles per interval: 15 m data reaches back
    to about 2026-09-09, 1 h to well before September.
  - `fundingHistory` returns 500 rows per page.
  - The order book and open interest are **live only**, with no history.
- **Book:** the live BTC spread is about 0.13 bp, with millions of dollars resting near the
  touch.

## 3. Architecture of the fork

```
tradingagents/
  dataflows/vendors/hyperliquid.py   HL /info client: paginated, cached, point-in-time slicing
  dataflows/vendors/crypto_pit.py    Fear & Greed (alternative.me) and point-in-time news (GDELT → Google News)
  agents/analysts/derivatives_analyst.py   funding / premium / volatility / carry analyst (replaces fundamentals)
  hyperliquid/
    venue.py        fee schedule, funding, slippage, margin constants (one source of truth)
    account.py      perp account: equity, position, fills, funding, liquidation (pure accounting)
    policy.py       rating → target exposure (long/short, long/flat, vol-target, optional stops)
    clock.py        decision cutoff, execution and settlement instants (the timing contract)
    runner.py       walk-forward loop: mark → settle memory → propagate → execute → hold
    baselines.py    buy & hold, time-series momentum, SMA cross, RSI reversion, random
    metrics.py      performance, Sharpe inference (Lo/Mertens), PSR, MinTRL, DSR, bootstrap, PT test
    report.py       tables (LaTeX/Markdown), figures, JSON summary
scripts/hl_backtest.py, scripts/hl_report.py, scripts/hl_leakage_probe.py
papers/  methods paper and empirical paper (LaTeX)
```

Upstream changes are opt-in and small. Each is gated on `config["crypto_venue"] == "hyperliquid"`:

1. The router learns a `hyperliquid` vendor for `get_stock_data` / `get_indicators`, the
   verified snapshot routes to it, and news gains point-in-time vendors.
2. The instrument context, which **every** agent reads, carries the venue contract: what a
   rating does, the costs, and that shorting is allowed.
3. The sentiment analyst swaps its live-only feeds for archived ones.
4. A `derivatives` analyst is added.
5. `settlement_mode="external"` stops the built-in Yahoo/SPY settlement, so the runner
   settles from simulator P&L.

Without the flag, behaviour is unchanged and the upstream tests still pass.

## 4. Timing contract (the core of "never see n+1")

For a decision dated **D** (UTC calendar day):

- **Information cutoff** `c_D = (D+1)T00:00:00Z`. Every input must be known strictly before
  `c_D`:
  - a candle only if its close time ≤ `c_D`;
  - funding prints with `time < c_D`;
  - Fear & Greed values with timestamp < `c_D`;
  - a news item only if its **known-at** time ≤ `c_D` (§6).
- **Execution** at `e_D = c_D + λ`, with λ = 15 min of agent latency. The fill is the 15 m
  bar open at `e_D`, worsened by half-spread plus slippage.
- **Holding** over `(e_D, e_{D+1}]`: the account is marked on 15 m bars and charged every
  hourly funding print.
- **Outcome of D** is known at `e_{D+1} = (D+2)T00:15Z`. So decision D+1 (cutoff
  `(D+2)T00:00Z`) must **not** see it, while decision D+2 may. The runner settles decision
  D−2 just before running D, and the memory filter (`resolution_date ≤ trade_date`) enforces
  the same rule independently.
- **Window:** decisions dated 2026-09-30 … 2026-10-29 (30 decisions). Executions run from
  2026-10-01 00:15 to 2026-10-30 00:15, and the P&L window is
  [2026-10-01 00:15, 2026-10-31 00:15] UTC, i.e. October, every period fully settled.

## 5. Accounting and costs (math to be derived in paper 1)

Notation: signed size q (BTC), mark price P, collateral C (USDC), equity E = C + q(P − P̄),
exposure x = qP/E.

- **Fill:** Δq at reference price P⁰ fills at P^f = P⁰(1 + sgn(Δq)·(s/2 + η)). The fee is
  φ|Δq|P^f (φ = 4.5 bp taker), and the slippage cost is |Δq|·|P^f − P⁰|.
- **Funding at hour h:** C ← C − q·P^o_h·f_h. The oracle is proxied by the 1 h open, an error
  of order premium × f, about 1e-9 of equity.
- **Daily return** on pre-trade equity:
  r_{D} ≈ x_D r^{BTC}_D − x_D Σ_h f_h − (φ + s/2 + η)|Δx_D| (+ second-order terms).
- **Break-even accuracy** for a daily sign strategy, with r ~ N(0, σ²), accuracy p and
  turnover E|Δx| = 1:
  p* = 1/2 + c / (2σ√(2/π)).
  With σ = 2.5%/day and c = 5.5 bp, p* ≈ 51.4%. Costs are a hurdle, not a wall, at daily
  frequency for BTC.
- **Power:** with T = 30 daily returns, SE(SR_daily) ≈ 1/√30 = 0.18, i.e. about 3.5
  annualized. An annualized Sharpe has to exceed about 6.8 to be significant at 5%, which
  corresponds to ~72% directional accuracy. **One month cannot establish skill.** The papers
  will say so and report intervals, not just point estimates.
- **Slippage default:** 1 bp per side. That is conservative against the measured
  0.1–0.5 bp, because the historical book is unobservable. Sensitivity: 0 / 2 / 5 bp, maker
  fills, fee tiers 0–6.

## 6. Look-ahead taxonomy and the control for each

| # | Leak | Control |
|---|---|---|
| L1 | future bars or values | slice by bar close time ≤ cutoff before computing any indicator; canary test: poison all data after the cutoff with absurd values and assert no tool output changes |
| L2 | imprecise publication stamps | known-at rule: GDELT `seendate` is used as is; Google News "midnight Pacific" stamps count as known 24 h later |
| L3 | selection and popularity measured today (upvotes, rankings) | StockTwits and Reddit are not used historically; Fear & Greed is used (archived daily); news-search ranking remains a residual risk |
| L4 | trading at prices seen before deciding | fill at `c_D + 15 min`, never at the close used for the signal |
| L5 | memory or reflection using unrealized outcomes | settle D−2 before D; resolution-date filter |
| L6 | LLM pretraining knows October 2026 | cutoffs: GPT-6 Sol 2026-04-20, Luna 2026-05-18 (secondary sources); empirical memorization probe on in-window vs pre-cutoff dates |
| L7 | real "today" leaking via the provider's system prompt | probe: ask the model for today's date with no date given |
| L8 | researcher degrees of freedom | this pre-registration; every variant reported; deflated Sharpe over the variants tried |

## 7. Pre-registered primary configuration

- **Universe and window:**
  - BTC perp on HL; $100,000 USDC starting equity, cross margin, max gross exposure 1×.
  - Decisions 2026-09-30 … 2026-10-29, cold-start memory.
- **Analysts:** market (HL candles), sentiment (point-in-time news, Fear & Greed, funding
  positioning), news (point-in-time), derivatives (funding, premium, volatility, carry).
  1 debate round, 1 risk round.
- **LLMs:** ChatGPT-plan provider, quick tier `gpt-6-luna`, deep tier `gpt-6-sol`.
- **Policy (long/short):** Buy +1.0, Overweight +0.5, Hold = keep, Underweight −0.5,
  Sell −1.0 of equity, times the PM's leverage multiple (see the amendment below).
  Rebalance band 5% of equity: drift smaller than that is not traded. No stops in the
  primary run.
- **Amendment A1 (2026-10-31, before any decision was run, at the user's request: "do not
  hesitate to use high leverage if you're sure"):**
  - The PM may set a leverage multiple L ∈ [1, 5]. The venue contract tells every agent to
    use it decisively when highly confident and to stay at 1× when the evidence is mixed.
  - Why the cap is 5× and not the venue's 40×: Kelly growth-optimal leverage is f* = μ/σ².
    With σ ≈ 1.7%/day and E|r| ≈ 1.35%/day, a 55%-accurate daily forecaster has
    μ ≈ 0.10 × 1.35% = 0.135%/day, so f* ≈ 4.7×; at 60% accuracy f* ≈ 9.3×.
  - At 40×, a 1.25% adverse move hits maintenance margin (1/L − m ≈ 1.25%).
  - The same decisions at 1× (multiple ignored) are reported as a counterfactual.
- **Costs:** tier 0 taker 4.5 bp, slippage 1 bp per side, hourly funding as printed, 15 min
  latency.
- **Primary metric:** net Sharpe (annualized √365) and net return vs buy-and-hold BTC perp
  under the same costs.
- **Secondary** (same decision sequence; counterfactual, because the agents saw primary-policy
  positions):
  - long/flat policy, vol-targeting (σ* = 40%, cap 2×), trader stop-losses;
  - cost sensitivity;
  - baselines.
- **Inference:**
  - Lo/Mertens Sharpe SE, PSR(0), MinTRL, DSR across the variants;
  - stationary-bootstrap CIs;
  - permutation test of timing (shuffle exposures, gross and net);
  - Pesaran–Timmermann directional test;
  - mean next-day return by rating, and rating/return rank IC.

## 8. Verification

- Unit tests:
  - accounting identities (ΔE = PnL − fees − funding, to machine precision);
  - funding sign conventions;
  - fee math;
  - policy map;
  - PSR / MinTRL against hand-computed values;
  - known-at rule;
  - point-in-time canary on every HL tool.
- The full upstream suite must still pass.
- Reproducibility: every HTTP response is cached under the run directory. Every decision,
  prompt digest, fill, funding payment and equity mark goes to a JSONL ledger, so the report
  rebuilds from the ledger alone.

## 9. Known limitations, stated up front

- One sample path per day: LLM sampling noise isn't averaged out. If plan quota allows, a
  subset of days is re-run to measure decision agreement.
- 30 observations: low power (§5).
- Slippage is modeled, not observed; there is no historical book.
- Open interest history isn't available from HL.
- News-search results are ranked today; headlines can be edited after publication.

## 10. Status after the run (2026-10-31, written after results)

- **Run:** all 30 decisions completed, 0 failures, about 154 s per decision. The ledger, the agent
  reports, the memory log and the data cache are in `experiments/hl_btc_2026-10`.
- **Result:** −0.53% net vs −8.13% for buy-and-hold; leverage was never used.
  - No detectable timing skill: direction right 14/30, PT p = 0.73.
  - Results and inference are in the empirical paper.
- **Departures from this plan:**
  - FRED macro data was off (no API key).
  - GDELT was rate-limited, so Google News with the known-at rule was used instead.
  - MACD was added to the baselines before the first decision.
- **Analyses added after results had been seen.** Each is labeled as such in the paper, and none changes
  a reported performance figure:
  - fixed-leverage sensitivity;
  - exposure/timing attribution (Lo 2008);
  - desk-agreement statistics;
  - forced and dense memorization probe;
  - consistency re-runs on 5 dates (planned in §9, run after).
