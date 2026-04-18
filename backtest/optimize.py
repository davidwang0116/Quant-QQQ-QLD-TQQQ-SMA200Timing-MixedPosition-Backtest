"""
Grid search to find optimal buy/sell thresholds for Combo and TQQQ strategies.

Run:
    python backtest/optimize.py
    python backtest/optimize.py --tranches 3
"""

from __future__ import annotations
import argparse
import logging
import sys
import itertools
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.downloader import fetch_prices
from strategies.engine import (
    StrategyConfig,
    build_price_series,
    strategy_timing_tqqq,
    strategy_combo,
)

logging.basicConfig(
    level=logging.WARNING,       # suppress INFO during grid search
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

END_DATE  = "2025-12-31"
START     = "2000-01-01"


def run_grid(tranches: int = 5, dip: float = -0.01) -> None:

    # ── Load data once ────────────────────────────────────────────
    print("Loading data...")
    qqq_full  = fetch_prices("QQQ",  start="1999-01-01", end=END_DATE)
    real_qld  = fetch_prices("QLD",  start="2006-06-21", end=END_DATE)
    real_tqqq = fetch_prices("TQQQ", start="2010-02-09", end=END_DATE)

    qqq, qld, tqqq = build_price_series(
        qqq_full=qqq_full, real_qld=real_qld, real_tqqq=real_tqqq,
        start=START, end=END_DATE,
    )

    # ── Parameter grid ────────────────────────────────────────────
    buy_range  = [round(x * 0.01 + 1.00, 2) for x in range(0, 20)]  # 1.00 ~ 1.20
    sell_range = [round(1.00 - x * 0.01, 2) for x in range(0, 30)]  # 1.00 ~ 0.70

    total      = len(buy_range) * len(sell_range)
    print(f"Grid: {len(buy_range)} buy × {len(sell_range)} sell = {total} combinations")
    print(f"Tranches={tranches}  Dip={dip*100:.1f}%\n")

    tqqq_results = []
    combo_results = []

    for i, (buy, sell) in enumerate(itertools.product(buy_range, sell_range)):
        if buy <= sell:          # buy threshold must be above sell threshold
            continue

        cfg = StrategyConfig(
            buy_threshold   = buy,
            sell_threshold  = sell,
            max_tranches    = tranches,
            dip_threshold   = dip,
            initial_capital = 100_000.0,
        )

        r_tqqq  = strategy_timing_tqqq(qqq, tqqq, cfg)
        r_combo = strategy_combo(qqq, qld, tqqq, cfg)

        tqqq_results.append({
            "buy": buy, "sell": sell,
            "total_ret":  round(r_tqqq.total_return() * 100, 2),
            "cagr":       round(r_tqqq.cagr() * 100, 2),
            "max_dd":     round(r_tqqq.max_drawdown() * 100, 2),
            "sharpe":     round(r_tqqq.sharpe(), 2),
        })
        combo_results.append({
            "buy": buy, "sell": sell,
            "total_ret":  round(r_combo.total_return() * 100, 2),
            "cagr":       round(r_combo.cagr() * 100, 2),
            "max_dd":     round(r_combo.max_drawdown() * 100, 2),
            "sharpe":     round(r_combo.sharpe(), 2),
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total} done...", end="\r")

    tqqq_df  = pd.DataFrame(tqqq_results).sort_values("total_ret", ascending=False)
    combo_df = pd.DataFrame(combo_results).sort_values("total_ret", ascending=False)

    # ── Print top 10 for each ─────────────────────────────────────
    _print_top(tqqq_df,  "Timing TQQQ — Top 10 by Total Return")
    _print_top(combo_df, "Combo 60/30/10 — Top 10 by Total Return")

    # ── Save full results ─────────────────────────────────────────
    out_dir = Path(__file__).parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    tqqq_df.to_csv(out_dir  / f"optimize_tqqq_t{tranches}.csv",  index=False)
    combo_df.to_csv(out_dir / f"optimize_combo_t{tranches}.csv", index=False)
    print(f"\nFull results saved to reports/optimize_*_t{tranches}.csv")

    # ── Heatmaps ──────────────────────────────────────────────────
    _plot_heatmap(tqqq_df,  "Timing TQQQ Total Return (%)",     tranches, "tqqq")
    _plot_heatmap(combo_df, "Combo 60/30/10 Total Return (%)",  tranches, "combo")


def _print_top(df: pd.DataFrame, title: str, n: int = 10) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Buy':>6}  {'Sell':>6}  {'Total Ret':>12}  {'CAGR':>8}  {'Max DD':>8}  {'Sharpe':>7}")
    print("-" * 72)
    for _, row in df.head(n).iterrows():
        print(
            f"  {row['buy']:>6.2f}  {row['sell']:>6.2f}  "
            f"{row['total_ret']:>11.2f}%  "
            f"{row['cagr']:>7.2f}%  "
            f"{row['max_dd']:>7.2f}%  "
            f"{row['sharpe']:>7.2f}"
        )
    print(sep)


def _plot_heatmap(df: pd.DataFrame, title: str, tranches: int, tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    pivot = df.pivot(index="sell", columns="buy", values="total_ret")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Total Return (%)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=45, fontsize=8, color="#c9d1d9")
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8, color="#c9d1d9")
    ax.set_xlabel("Buy Threshold", color="#8b949e")
    ax.set_ylabel("Sell Threshold", color="#8b949e")
    ax.set_title(f"{title}  |  Tranches={tranches}", color="#c9d1d9", fontsize=12)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # Annotate best cell
    best_idx = df["total_ret"].idxmax()
    best_buy  = df.loc[best_idx, "buy"]
    best_sell = df.loc[best_idx, "sell"]
    col_idx = list(pivot.columns).index(best_buy)
    row_idx = list(pivot.index).index(best_sell)
    ax.plot(col_idx, row_idx, "w*", markersize=14, label=f"Best: buy={best_buy} sell={best_sell}")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    plt.tight_layout()
    out = Path(__file__).parent.parent / "reports" / f"optimize_{tag}_t{tranches}_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Heatmap saved → {out.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search for optimal buy/sell thresholds")
    p.add_argument("--tranches", type=int,   default=5,     help="DCA tranches (default 5)")
    p.add_argument("--dip",      type=float, default=-0.01, help="Dip threshold (default -0.01)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_grid(tranches=args.tranches, dip=args.dip)