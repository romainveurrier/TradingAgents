# TradingAgents-HL v2: from storyteller to scientist

**Goal:** a materially better BTC trading system on Hyperliquid, using leverage as aggressively as
the mathematics allows, and evaluated so that "better" is real rather than an artifact of luck or
leakage.

Written 2026-10-31 after the October run (v1) and a 3.5-year study of the venue's own data. Every
number below is measured on Hyperliquid BTC from 2023-05 to 2026-10, unless marked otherwise.

---

## 1. Diagnosis: why v1 could not win

| Symptom in October | Root cause | v2 fix |
|---|---|---|
| Direction right 14/30; timing contributed −1.98% | LLMs were asked to forecast daily direction from public, lagging, already-priced information (technicals, headlines) | Stop asking LLMs for direction. Use them where text carries information: events, regimes, vetoes (§4.4, §4.6) |
| Ratings were ordinal (Buy…Sell); sizing was arbitrary | No probabilities, so no edge estimate and no rational size | Every component emits a **predictive distribution**, so sizing follows the edge (§5) |
| The PM never levered; the user wanted leverage | Leverage was a free-text whim, not a function of measured edge | Leverage is computed, never chosen: Kelly with parameter uncertainty and a liquidation constraint (§5) |
| Reflection memory did not improve decisions | Prose lessons, no measurement of who was right | A **forecasting tournament**: proper scoring rules, per-agent skill weights (§4.5) |
| One evaluation month: no power | Sharpe SE ≈ 3.5 at T = 30 | Multi-year quant backtests (CPCV), LLM tests only after their measured knowledge horizon, ≥ 3 months forward testing (§7) |

### What the long history says (2023-05 → 2026-10, net of 5.5 bp per side and funding)

| Strategy | Ann. return | Vol | Sharpe | Max DD | Kelly L* = μ/σ² |
|---|---|---|---|---|---|
| Buy & hold perp | +27.0% | 45.5% | 0.59 | 54.9% | 1.3× |
| TSMOM 10 / 20 / 30 / 60 / 90 / 120 d | −22% … +15% | ~45% | −0.49 … +0.33 | 51–70% | ≤ 0.7× |
| TSMOM 30 d, vol-targeted 40% | +18.6% | 44.0% | 0.42 | 57.3% | 1.0× |
| **Carry: long spot + short perp** | **+13.9%** | **~1.0%*** | **very high*** | **1.1%** | **≫ 10×*** |

\* Daily aggregation of hourly-average premia understates intraday basis risk. Treat the carry
Sharpe as an upper bound to be re-measured on tick data.

- Funding by year: 2023 +13.9%, 2024 +24.1%, 2025 +10.6%, 2026 +5.4%. It was negative on 12% of
  days.
- October's +9% for TSMOM(30) was the regime. Over 3.5 years the same rule has a Sharpe of 0.15.

### Leverage, measured rather than modeled

The table counts intraday adverse excursions from the open that reach the liquidation distance
d(L) = (1/L − m)/(1 − m), with m = 1.25%:

| L | Liquidating move | Days it happened (long) | Liquidations per year |
|---|---|---|---|
| 5× | 19.0% | 0.00% | ~0 |
| 10× | 8.9% | 0.87% | ~3 |
| 20× | 3.8% | 11.1% | ~40 |
| 40× | 1.27% | 47.9% | ~175 |

There are two consequences.

1. **Directional BTC at 20–40× is not a high-reward strategy. It is a near-certain wipe-out.** Beyond
   2·L* the expected log-growth is negative, so above Kelly more leverage means *less* long-run
   wealth and *more* risk at the same time.
2. **High leverage belongs on high-Sharpe, low-volatility streams.** That is the carry and basis
   engine, not the directional bet. The plan therefore puts the leverage where it pays.

---

## 2. Design principles

1. **Edge first, leverage second.** Leverage multiplies an edge (and its costs). It never creates one.
2. **Distributions, not opinions.** Every model outputs P(r_{t+h} | info_t): mean, volatility,
   quantiles and P(up).
3. **LLMs for language, quant for numbers.** LLMs read text, detect regimes and veto. Numeric
   forecasting and sizing are statistical.
4. **Bitemporal truth.** Every datum has an *event time* and a *knowledge time*. Generalizing v1's
   known-at rule, nothing is used before its knowledge time.
5. **Pre-register, count every trial, deflate.** The research process itself is the main overfitting
   risk.

---

## 3. Data platform (phase 0)

| Source | Content | History | Use |
|---|---|---|---|
| Hyperliquid `/info` | 1d/1h/15m candles, hourly funding and premium | 2023 → | prices, carry, the funding model |
| Hyperliquid websocket (**record forward**) | L2 book, trades, open interest, mark/oracle | from start of recording | microstructure, crowding, execution |
| Binance public data (`data.binance.vision`) | klines, funding, aggTrades, futures metrics (OI, long/short ratios) | 2019 → | long-horizon research, cross-venue basis and lead-lag |
| Deribit public API | DVOL implied-volatility index, options | 2021 → | implied vs realized vol, variance risk premium, tail regime |
| US spot BTC ETF daily flows | creations and redemptions | 2024 → | flow signal, with a publication-lag known-at |
| News (Google News RSS + any timestamped feed) | headlines with known-at | rolling | the LLM event engine |

A **bitemporal feature store** (Parquet, keyed by `(feature, event_time, known_at)`) extends the
canary test to every feature: poison everything after the cutoff and assert identical features.
Sources and their exact API capabilities must be verified at build time (e.g. which Binance metrics
files exist, and for which dates).

---

## 4. Return engines

Each engine is developed, pre-registered and evaluated *separately*, then combined (§4.5).

### 4.1 E1: Funding-carry engine (market-neutral; the leverage engine)

**Trade.** Short the BTC perp, hedge the delta with spot BTC. P&L ≈ Σ funding − Δ basis − costs.

**New: a clamp-band funding model for Hyperliquid.** From the venue rule (methods paper,
Prop. 2), F = I whenever the premium P ∈ [I − c, I + c]; otherwise F = P ∓ c.
- Model the hourly premium as an Ornstein–Uhlenbeck process, dP = κ(θ − P)dt + η dW, with θ
  conditioned on OI changes, the cross-venue basis and volatility.
- Funding *departures* are then first-passage events out of the band, with closed-form or simulated
  probabilities.
- This yields E[F over the next 24–72 h] and the probability of negative funding.

**Policy.** Hold the carry when E[F_h] − entry and exit costs/h > threshold. De-lever ahead of
predicted band exits to the negative side. Shrink in high-volatility regimes, where basis gaps
widen.

**Leverage.** The stream's volatility is very low, so Kelly supports high leverage. The binding
constraints are structural:
1. Funding the spot leg. Spot is fully paid unless Hyperliquid spot or portfolio margin, or a second
   venue, finances it. **Verify venue capabilities first.** Without them, the notional per unit of
   equity is ≈ L/(L + 1) < 1×.
2. Liquidation of the short leg in a squeeze. It is margined separately unless portfolio margin
   nets the legs. Apply the liquidation-quantile cap of §5 to the leg.
3. Tail risks: funding regime shifts, basis blowouts, venue, oracle and stablecoin risk.

**Target.** 2–5× notional per unit of equity where the margin structure allows it.

| Funding regime | Expected return at 3× |
|---|---|
| 2024-like (24%/yr) | ≈ 40–70% a year |
| 2026-like (5.4%/yr) | ≈ 10–15% a year |

These figures are before costs and incidents, and are hypotheses to be tested.

### 4.2 E2: Robust trend engine (directional, small Kelly)

- **No single tuned lookback,** because the 3.5-year table shows lookback roulette. Use an ensemble
  of 20–120 d signals, each scaled by its t-statistic of slope, in the spirit of Hurst, Ooi and
  Pedersen 2017.
- **Volatility management:** a HAR-RV forecast (Corsi 2009) sets the risk budget. Add DVOL when
  implied volatility exceeds realized: a regime with tail risk priced in.
- **Leverage:** Bayesian-Kelly ≤ 1–1.5×, from §5. History says this is the ceiling.

### 4.3 E3: Positioning and squeeze engine (short horizon, conditional)

- **Signals:**
  - funding departures above the band;
  - the OI build-up rate;
  - long/short account skews;
  - premium dislocations.
- **Hypothesis:** extreme one-sided positioning predicts *asymmetric* next-24 h moves against the
  crowd (squeeze risk), so it is used as a skew/veto signal more than a direction signal.
- **Data:** long history comes from Binance; Hyperliquid OI is recorded forward.

### 4.4 E4: LLM event engine (sparse, high-conviction)

- **Turn the desk into an event detector.** On each new headline batch, an LLM extracts structured
  events:
  - type (macro print, Fed, ETF flow shock, hack, regulation, exchange incident, large holder);
  - direction and surprise relative to expectation;
  - credibility;
  - a calibrated probability distribution of the next 1–24 h move.
- **Evidence base:** LLM news scores predict short-horizon equity returns (Lopez-Lira & Tang 2023).
  Event studies on our own archive estimate impact distributions per event type.
- **Trades are rare and event-triggered,** not a daily opinion. The edge, if any, is in the minutes
  to hours after a surprise, not in a daily reprise of the 10-day EMA.

### 4.5 Combining the engines: a forecasting tournament

- Every agent and engine emits P(up over h) and a quantile forecast, scored with **strictly proper
  rules**: the log score and CRPS (Gneiting & Raftery 2007).
- **Skill weights.** Weight each agent with exponentially weighted Hedge, w_i ∝ exp(−η · cumulative
  log loss_i) (Freund & Schapire 1997). The desk learns *whom to trust* from outcomes, not prose
  reflections.
- **Pooling.** Pool with **extremized logit aggregation**,
  logit p̂ = a · Σ w_i logit p_i with a > 1 (Satopää et al. 2014). This corrects the under-confidence
  of averaged forecasters, as in the superforecasting literature.
- **Calibration.** Apply isotonic calibration on walk-forward data before any probability reaches the
  sizer.

### 4.6 LLM as meta-labeler (new combination)

- Quant engines (E2, E3) propose a side. An LLM **meta-labeler** (López de Prado 2018, ch. 3) reads
  the text context and outputs P(the proposed trade is profitable).
- The size scales with that probability, and a veto applies below a threshold.
- **Success criterion:** a lift in precision and Sharpe of the primary signal on out-of-sample data,
  measured against an identical meta-labeler fed no text.

---

## 5. Sizing and leverage: "as high as the math allows"

**Liquidation-quantile Bayesian Kelly (LQK, new).** For each engine and holding horizon h, the
leverage is the smallest of three caps:

1. **Growth cap (Bayesian Kelly).** L_kelly = c_K · μ̂ / (σ̂² + s²_μ), where s²_μ is the posterior
   variance of the edge (Browne & Whitt 1996). Uncertainty about μ shrinks leverage automatically.
   c_K = 1 is the growth-optimal "aggressive" profile; anything above 1 is dominated.
2. **Liquidation cap.** Require d(L) ≥ MAE_{1−ε,h}, the (1 − ε) quantile of the maximum adverse
   excursion over h, fitted with EVT peaks-over-threshold on GARCH-filtered returns (McNeil & Frey
   2000). This gives L_liq = 1 / (m + (1 − m) · MAE_{1−ε,h}). The empirical table in §1 shows the
   scale: about 7–8× at a daily 99.9% quantile.
3. **Drawdown cap.** Exposure is proportional to the distance from a drawdown floor (Grossman & Zhou
   1993), with a risk-constrained Kelly bound P(W ever < αW₀) ≤ β (Busseti, Ryu & Boyd 2016).

**Cost-aware growth.** g(L) = L(μ − τc) − ½L²σ², where τ is daily turnover and c the cost per side.
At L = 10 with full daily turnover, taker fees alone cost 365 · 10 · 5.5 bp ≈ **200% of equity a
year**. High leverage is viable only with **maker execution** and low turnover. That is why the
carry engine (low turnover) can be levered and a daily directional flipper cannot.

**Profiles.**

| Profile | c_K | ε (liquidation per year) | Drawdown floor |
|---|---|---|---|
| Aggressive | 1.0 (full Kelly) | 1% | 50% |
| Balanced | 0.5 | 0.1% | 25% |

Every profile is capped at the venue's maximum of 40× and below by the three formulas. The
aggressive profile is the most aggressive *rational* setting; we will also report, never deploy,
what 2× Kelly and the venue maximum would have done.

---

## 6. Execution

- **Maker-first.** Post limit orders inside the spread with a fill-probability model, falling back
  to taker after a timeout. Model adverse selection: the post-fill drift of filled maker orders.
- **Fee tiers.** Plan the volume needed for tier 1 and above. HYPE staking discounts of 5–40% are a
  direct lever on the cost of high leverage.
- **Venue-native protective orders.** Stop-market and take-profit orders resting on the venue
  protect leveraged legs without depending on our latency.
- **Order-flow imbalance** (Cont, Kukanov & Stoikov 2014) to time entries, not as alpha.

---

## 7. Evaluation science (what makes "better" credible)

1. **Quant engines:** 2019–2026 backtests (Binance history plus Hyperliquid where available), with
   combinatorial purged cross-validation and embargo (López de Prado 2018).
   - Report the probability of backtest overfitting (Bailey et al. 2017).
   - Report Hansen's SPA / White's Reality Check against the trial registry.
   - Report the DSR over **every configuration ever tried** (the registry is append-only).
2. **LLM components: knowledge-horizon estimation (KHE, new).**
   - Run the forced-recall probe monthly, and find the change-point where recall error jumps.
   - v1 found GPT-6 Sol recalls prices to within 0.1% until January 2026 and loses them from
     February, about three months *before* its advertised cutoff.
   - LLM-in-the-loop backtests run only after the measured horizon, or on **anonymized inputs**
     (indexed prices, masked dates and names; Glasserman & Lin 2023). We verify with the same probe
     that anonymization removes recall.
3. **Forward test.** At least 3 months of paper trading on Hyperliquid testnet, or mainnet at minimum
   size, with live data. Go/no-go thresholds are fixed now:
   - PSR(SR > 0) ≥ 0.95;
   - DSR ≥ 0.90;
   - maximum drawdown ≤ the profile floor;
   - zero liquidations.
4. **Power.** The track record needed is MinTRL ≈ 1 + (z/SR_daily)² days:
   - about 8 months for an annualized Sharpe of 2;
   - about 2.7 years for an annualized Sharpe of 1.

   The evaluation length follows from the claimed Sharpe, not from patience.
5. **Agent sampling.** Draw K ≥ 3 samples per decision and evaluate the policy distribution. v1's
   5/5 re-run agreement is a lower bound, possibly due to anchoring.

---

## 8. Roadmap

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| 0 | Bitemporal store; Binance, Deribit and ETF loaders; live Hyperliquid recorder; canary test on every feature | All canaries pass; data audit documented |
| 1 | **E1 carry** with the clamp-band funding model, plus the venue-capability check for spot/portfolio margin | OOS carry Sharpe and MDD on CPCV; funding-forecast log score beats persistence |
| 2 | E2 robust trend + HAR/DVOL risk; LQK sizer | DSR ≥ 0.9 over the registry; LQK keeps simulated liquidations ≤ ε |
| 3 | E4 LLM event engine + E3 positioning; tournament pooling; LLM meta-labeler | Proper-score improvement and meta-label lift **after the knowledge horizon** |
| 4 | Maker execution engine; testnet forward run of the combined book | Forward go/no-go thresholds (§7.3) |
| 5 | Minimum-size live deployment with a kill switch, max daily loss and venue-incident playbook | Forward results within the predicted distribution |

**First step:** Phases 0–1. The carry engine is the only measured high-Sharpe stream in the data,
and the only place where high leverage is mathematically justified. It needs no LLM, so it can be
validated on years of history without pretraining leakage.

---

## 9. References

- Busseti, E., Ryu, E. K., Boyd, S. (2016). Risk-constrained Kelly gambling. *Journal of Investing*.
- Browne, S., Whitt, W. (1996). Portfolio choice and the Bayesian Kelly criterion. *Advances in
  Applied Probability* 28(4).
- Cont, R., Kukanov, A., Stoikov, S. (2014). The price impact of order book events. *Journal of
  Financial Econometrics* 12(1).
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of
  Financial Econometrics* 7(2).
- Freund, Y., Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning.
  *Journal of Computer and System Sciences* 55(1).
- Glasserman, P., Lin, C. (2023). Assessing look-ahead bias in stock return predictions generated by
  GPT sentiment analysis. arXiv:2309.17322.
- Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation.
  *JASA* 102.
- Grossman, S. J., Zhou, Z. (1993). Optimal investment strategies for controlling drawdowns.
  *Mathematical Finance* 3(3).
- Hansen, P. R. (2005). A test for superior predictive ability. *JBES* 23(4).
- Hurst, B., Ooi, Y. H., Pedersen, L. H. (2017). A century of evidence on trend-following investing.
  *Journal of Portfolio Management* 44(1).
- Koijen, R., Moskowitz, T., Pedersen, L. H., Vrugt, E. (2018). Carry. *Journal of Financial
  Economics* 127(2).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lopez-Lira, A., Tang, Y. (2023). Can ChatGPT forecast stock price movements? arXiv:2304.07619.
- McNeil, A. J., Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic
  financial time series: an extreme value approach. *Journal of Empirical Finance* 7.
- Satopää, V. A., et al. (2014). Combining multiple probability predictions using a simple logit
  model. *International Journal of Forecasting* 30(2).
- White, H. (2000). A reality check for data snooping. *Econometrica* 68(5).
- Earlier references: see `papers/references.bib`.
