# TQQQ MA200 Strategy Backtest
# TQQQ MA200 牛熊择时策略回测框架

A production-grade quantitative backtest framework for a QQQ 200-day moving average bull/bear timing strategy applied to TQQQ, featuring tranche-based position building and multi-period evaluation.

基于 QQQ 200 日均线牛熊判断的量化回测框架，交易标的为 TQQQ，支持分批建仓与多时间段评估。

---

## Strategy Logic / 策略逻辑

### Signal Rules / 信号规则

| Signal / 信号 | Condition / 条件 | Action / 操作 |
|---|---|---|
| Bull confirmed / 牛市确认 | `QQQ > MA200 × 1.04` | Enter TQQQ in tranches / 分批买入 TQQQ |
| Bear confirmed / 熊市确认 | `QQQ < MA200 × 0.97` | Full exit → Cash / 一次性清仓 |
| In-between / 区间震荡 | Neither threshold crossed / 未触及阈值 | Hold current position / 保持现状 |

### Entry Method / 入场方式

| Parameter / 参数 | Behaviour / 说明 |
|---|---|
| Each tranche / 每批仓位 | `1 / max_tranches` of current total equity / 当前总权益的 `1/max_tranches` |
| Final tranche / 最后一批 | Invest all remaining cash / 用完全部剩余现金 |
| Max tranches / 最大批次 | Configurable via `--tranches` / 通过 `--tranches` 参数控制 |
| Trigger / 触发条件 | QQQ daily return ≤ −1% while in bull zone / 牛市区间内 QQQ 当日跌幅 ≥ 1% |

**Example / 示例：**
- `--tranches 1` → Full position on first signal / 第一个信号直接全仓
- `--tranches 5` → ~20% of equity per dip, up to 5 entries / 每次回调买入约 20% 权益，最多 5 批

### Exit Method / 出场方式

When `QQQ < MA200 × 0.97`, all TQQQ shares are sold in a single trade and the tranche counter resets to zero.

当 `QQQ < MA200 × 0.97` 时，一次性卖出所有 TQQQ 持仓，批次计数归零。

---

## Project Structure / 项目结构

```
quant_backtest/
├── strategies/
│   └── tqqq_ma200.py       # Strategy logic, signal generator, metrics
│                           # 策略核心：信号生成、回测引擎、指标计算
├── data/
│   ├── downloader.py       # yfinance wrapper with Parquet cache
│   │                       # yfinance 下载器，自动 Parquet 缓存
│   └── cache/              # Auto-generated, git-ignored / 自动生成，已忽略
├── backtest/
│   └── run.py              # CLI runner + multi-period table + chart
│                           # CLI 入口、多时间段评估表、图表生成
├── reports/
│   └── backtest_result.png # Latest chart, committed by CI / 最新图表
├── tests/
│   └── test_strategy.py    # Unit tests (pytest) / 单元测试
├── .github/
│   └── workflows/
│       └── ci.yml          # Lint → Test → Weekly backtest / 自动化流水线
├── requirements.txt
├── pyproject.toml          # ruff + mypy + pytest config
└── .gitignore
```

---

## Quick Start / 快速开始

```bash
# 1. Create conda environment / 创建 conda 环境
conda create -n quant python=3.11 -y
conda activate quant

# 2. Install dependencies / 安装依赖
pip install -r requirements.txt

# 3. Run backtest / 运行回测
python backtest/run.py

# 4. Custom parameters / 自定义参数
python backtest/run.py --buy 1.05 --sell 0.96 --tranches 3 --capital 50000

# 5. Force re-download data / 强制重新下载数据
python backtest/run.py --refresh

# 6. Run tests / 运行测试
pytest tests/ -v

# 7. Lint and format / 代码检查与格式化
ruff check .
ruff format .
```

---

## CLI Options / 命令行参数

| Flag | Default | Description / 说明 |
|------|---------|---|
| `--buy` | `1.04` | Buy threshold multiplier / 买入阈值倍数（QQQ > MA200 × this）|
| `--sell` | `0.97` | Sell threshold multiplier / 卖出阈值倍数（QQQ < MA200 × this）|
| `--ma` | `200` | MA period in days / 均线周期（天）|
| `--tranches` | `5` | Entry tranches: 1 = all-in, N = 1/N equity per tranche / 建仓批次：1=全仓，N=每批 1/N 权益 |
| `--capital` | `100000` | Initial capital in USD / 初始资金（美元）|
| `--refresh` | `False` | Force re-download price data / 强制重新下载数据 |

---

## GitHub Setup / 推送到 GitHub

```bash
git init
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git add .
git commit -m "feat: initial TQQQ MA200 backtest framework"
git branch -M main
git push -u origin main
```

The CI pipeline triggers on every push and runs a full backtest every Monday at 02:00 UTC, committing the updated chart back to the repository.

CI 流水线在每次 push 时自动触发，并在每周一 UTC 02:00 执行完整回测，将最新图表 commit 回仓库。

---

## Risk Warning / 风险提示

TQQQ is a 3× leveraged ETF subject to volatility decay. Past backtest results do not guarantee future performance. This project is for educational and research purposes only — **not investment advice**.

TQQQ 为 3 倍杠杆 ETF，存在波动率损耗（volatility decay）。历史回测结果不代表未来收益。本项目仅供学习与研究用途，**不构成任何投资建议**。

---

## Credits / 致谢

Strategy concept inspired by the following YouTube video:
策略思路部分参考自以下 YouTube 视频：

> [别再定投 QQQ，这个混合配方才是普通人投资的终极答案！26 年回測結果太驚人](https://www.youtube.com/watch?v=ey7B8NthhpM)

This project is an independent implementation. The backtest logic, code, and data are unrelated to the original video and are intended for educational purposes only.

本项目为独立实现，回测逻辑、代码与数据均与原视频无关，仅供学习研究用途。