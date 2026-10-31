# Papers

| PDF | What it covers |
|---|---|
| [`TradingAgents-HL_methods.pdf`](TradingAgents-HL_methods.pdf) | How TradingAgents was adapted to Hyperliquid BTC perps. Covers fees, the funding regime (verified on every October print), exact accounting and liquidation, break-even accuracy, Kelly and leverage, the timing contract, the look-ahead taxonomy and its controls, and the statistical protocol and its power. |
| [`TradingAgents-HL_october-2026.pdf`](TradingAgents-HL_october-2026.pdf) | The walk-forward run over October 2026: what the desk did, net performance against baselines, skill tests, robustness, the leakage audit, sampling noise, and case studies. |

Every number, table and figure is generated from the run directory:

```bash
make -C papers all      # = scripts/hl_report.py on experiments/hl_btc_2026-10, then latexmk on both papers
```

`generated/` holds the macros (`numbers.tex`), tables and `summary.json`; `figures/` holds the plots. The
leakage-probe and consistency files come from `scripts/hl_leakage_probe.py` and `scripts/hl_rerun_consistency.py`.
