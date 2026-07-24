"""Run backtests with different MA periods and compare results.

Supports QQQ (Nasdaq-100) and SPY (S&P 500) markets.

Usage:
    python backtest/compare_ma.py --market qqq --ma 200 225 300 325
    python backtest/compare_ma.py --market spy --ma 200 225 300
    python backtest/compare_ma.py --market both --ma 200 225 300 325
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

MARKETS = {
    "qqq": {
        "script":     "backtest/run.py",
        "results_md": "reports/results.md",
        "label":      "QQQ (Nasdaq-100)",
        "strategies": ["Timing TQQQ", "Combo 60/30/10"],
        "tag":        "tqqq",
    },
    "spy": {
        "script":     "backtest/run_spy.py",
        "results_md": "reports/spy/spy_results.md",
        "label":      "SPY (S&P 500)",
        "strategies": ["Timing UPRO", "Combo SPY 60/30/10"],
        "tag":        "upro",
    },
}

PERIODS = ["Full History", "Last 15 Years", "Last 10 Years", "Last  5 Years"]
PERIOD_LABELS = {
    "Full History": "Full History (~26Y)",
    "Last 15 Years": "Last 15 Years",
    "Last 10 Years": "Last 10 Years",
    "Last  5 Years": "Last 5 Years",
}


def parse_all_tables(text: str) -> dict[str, list[dict]]:
    """Parse all period performance tables from results.md."""
    lines = text.splitlines()
    tables: dict[str, list[dict]] = {}
    current_period: str | None = None
    headers: list[str] = []
    rows: list[dict] = []

    for line in lines:
        matched = False
        for period in PERIODS:
            if f"### {period}" in line:
                if current_period and rows:
                    tables[current_period] = rows
                current_period = period
                headers = []
                rows = []
                matched = True
                break
        if matched:
            continue

        if current_period and "|" in line:
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]
            if not cells:
                continue
            if "Strategy" in cells[0]:
                headers = cells
                continue
            if cells[0].startswith("---"):
                continue
            if headers and len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))

    if current_period and rows:
        tables[current_period] = rows

    return tables


def run_backtest(market: str, ma: int) -> dict[str, list[dict]]:
    """Run backtest for given market and MA period, return parsed results."""
    cfg = MARKETS[market]
    script = PROJECT_ROOT / cfg["script"]
    results_md = PROJECT_ROOT / cfg["results_md"]

    print(f"\n{'='*60}")
    print(f"  [{cfg['label']}] Running --ma {ma}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, str(script), "--ma", str(ma)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )
    if result.returncode != 0:
        print(f"ERROR: backtest failed for {market} MA={ma}")
        print(result.stderr[-500:])
        return {}

    text = results_md.read_text(encoding="utf-8")
    tables = parse_all_tables(text)

    # Print summary for the primary strategy
    primary = cfg["strategies"][0]
    for period in PERIODS:
        if period in tables:
            match = [r for r in tables[period] if primary in r.get("Strategy", "")]
            if match:
                r = match[0]
                print(f"  {PERIOD_LABELS[period]:25s}  {primary} CAGR={r.get('CAGR', '-'):>8s}  Sharpe={r.get('Sharpe', '-'):>5s}")
    return tables


def build_report(
    market: str,
    all_results: dict[int, dict[str, list[dict]]],
) -> list[str]:
    """Build markdown report lines for one market."""
    cfg = MARKETS[market]
    strategies = cfg["strategies"]
    ma_list = sorted(all_results.keys())

    report: list[str] = []
    report.append(f"# {cfg['label']} — MA Period Comparison Report\n")
    report.append(f"**Parameters:** Buy `×1.03` | Sell `×0.83` | Tranches `5` | Dip `-1.0%` | Capital `$100,000`\n")
    report.append(f"**MA periods tested:** {', '.join(str(m) for m in ma_list)}\n")
    report.append("---\n")

    for strategy in strategies:
        report.append(f"## {strategy}\n")

        for period in PERIODS:
            label = PERIOD_LABELS[period]
            report.append(f"### {label}\n")
            report.append("| MA | Total Return | Final Value | CAGR | Max DD | Sharpe | In Market |")
            report.append("|---:|---:|---:|---:|---:|---:|---:|")

            best_cagr_ma = None
            best_cagr_val = -999.0

            for ma in ma_list:
                tables = all_results[ma]
                rows = tables.get(period, [])
                match = [r for r in rows if strategy in r.get("Strategy", "")]
                if not match:
                    report.append(f"| {ma} | - | - | - | - | - | - |")
                    continue
                r = match[0]
                tr = r.get("Total Return", "-")
                fv = r.get("Final Value", "-")
                cagr = r.get("CAGR", "-")
                dd = r.get("Max DD", "-")
                sharpe = r.get("Sharpe", "-")
                in_mkt = r.get("In Market", "-")
                report.append(f"| {ma} | {tr} | {fv} | {cagr} | {dd} | {sharpe} | {in_mkt} |")

                try:
                    val = float(cagr.replace("%", "").replace("+", ""))
                    if val > best_cagr_val:
                        best_cagr_val = val
                        best_cagr_ma = ma
                except ValueError:
                    pass

            if best_cagr_ma is not None:
                report.append(f"\n**Best CAGR:** MA={best_cagr_ma} ({best_cagr_val:+.2f}%)\n")
            report.append("")

    # ── Conclusion ──
    report.append("---\n")
    report.append("## Conclusion / 结论\n")

    primary = strategies[0]
    best_ma = None
    best_val = -999.0
    for ma, tables in all_results.items():
        rows = tables.get("Full History", [])
        match = [r for r in rows if primary in r.get("Strategy", "")]
        if match:
            try:
                val = float(match[0]["CAGR"].replace("%", "").replace("+", ""))
                if val > best_val:
                    best_val = val
                    best_ma = ma
            except (ValueError, KeyError):
                pass

    if best_ma is not None:
        report.append(f"**Best MA period (by {primary} full-history CAGR): MA={best_ma}** (CAGR {best_val:+.2f}%)\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Compare backtest results across MA periods")
    parser.add_argument("--market", type=str, default="qqq",
                        choices=["qqq", "spy", "both"],
                        help="Market to backtest: qqq, spy, or both (default: qqq)")
    parser.add_argument("--ma", nargs="+", type=int, default=[200, 225, 300, 325],
                        help="MA periods to compare (default: 200 225 300 325)")
    args = parser.parse_args()

    markets = ["qqq", "spy"] if args.market == "both" else [args.market]

    # Collect all results in memory: {market: {ma: tables}}
    all_market_results: dict[str, dict[int, dict[str, list[dict]]]] = {}

    for market in markets:
        cfg = MARKETS[market]
        print(f"\n{'#'*60}")
        print(f"  Market: {cfg['label']}")
        print(f"{'#'*60}")

        all_results: dict[int, dict[str, list[dict]]] = {}
        for ma in args.ma:
            tables = run_backtest(market, ma)
            if tables:
                all_results[ma] = tables

        if not all_results:
            print(f"No results collected for {cfg['label']}!")
            continue

        all_market_results[market] = all_results

        report = build_report(market, all_results)

        out_path = PROJECT_ROOT / "reports" / f"ma_comparison_{market}.md"
        out_path.write_text("\n".join(report), encoding="utf-8")
        print(f"\n✅ Comparison report saved → {out_path}")

    # If both markets, also generate a cross-market summary from in-memory data
    if args.market == "both" and len(all_market_results) == 2:
        _build_cross_market_summary(args.ma, all_market_results)


def _build_cross_market_summary(
    ma_list: list[int],
    all_market_results: dict[str, dict[int, dict[str, list[dict]]]],
) -> None:
    """Generate a side-by-side QQQ vs SPY comparison for each MA."""
    report: list[str] = []
    report.append("# QQQ vs SPY — Cross-Market MA Comparison\n")
    report.append(f"**Parameters:** Buy `×1.03` | Sell `×0.83` | Tranches `5` | Dip `-1.0%` | Capital `$100,000`\n")
    report.append(f"**MA periods:** {', '.join(str(m) for m in ma_list)}\n")
    report.append("---\n")

    for ma in sorted(ma_list):
        report.append(f"## MA={ma}\n")

        for period in PERIODS:
            label = PERIOD_LABELS[period]
            report.append(f"### {label}\n")
            report.append("| Market | Strategy | Total Return | CAGR | Max DD | Sharpe |")
            report.append("|--------|----------|---:|---:|---:|---:|")

            for market in ["qqq", "spy"]:
                cfg = MARKETS[market]
                market_data = all_market_results.get(market, {})
                tables = market_data.get(ma, {})
                rows = tables.get(period, [])

                for strategy in cfg["strategies"]:
                    match = [r for r in rows if strategy in r.get("Strategy", "")]
                    if match:
                        r = match[0]
                        report.append(
                            f"| {cfg['label']} | {strategy} "
                            f"| {r.get('Total Return', '-')} "
                            f"| {r.get('CAGR', '-')} "
                            f"| {r.get('Max DD', '-')} "
                            f"| {r.get('Sharpe', '-')} |"
                        )
            report.append("")

    out_path = PROJECT_ROOT / "reports" / "ma_comparison_cross_market.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\n✅ Cross-market comparison saved → {out_path}")


if __name__ == "__main__":
    main()
