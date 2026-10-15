"""End-to-end: CSVs on disk -> CLI backtest -> report files. Synthetic data; checks plumbing only."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import breakout_series, ohlcv_from_close

ROOT = Path(__file__).resolve().parent.parent


def _write(path: Path, df: pd.DataFrame) -> None:
    df.rename_axis("date").reset_index().to_csv(path, index=False)


def test_cli_backtest_end_to_end(tmp_path):
    daily = tmp_path / "daily"
    daily.mkdir()
    n = 3000
    dates = pd.bdate_range("2012-01-02", periods=n)
    _write(daily / "NIFTY 50.csv", ohlcv_from_close(np.linspace(5000, 25000, n), dates))
    rng = np.random.default_rng(1)
    for k in range(5):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.02, n)))
        df = ohlcv_from_close(c, dates, vol=5e6)
        _write(daily / f"S{k}.csv", df)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text((ROOT / "config" / "strategy.yaml").read_text()
                        .replace("min_rs_percentile: 70", "min_rs_percentile: 0")
                        .replace("max_atr_contraction: 1.0", "max_atr_contraction: 10"))
    out = tmp_path / "reports"
    r = subprocess.run([sys.executable, "-m", "swing.cli", "--config", str(cfg_path), "--data-dir", str(tmp_path),
                        "--out", str(out), "backtest"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    run_dir = next(out.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert set(summary) == {"full", "in_sample", "out_of_sample"}
    assert "Verdict" in (run_dir / "report.md").read_text()

    r = subprocess.run([sys.executable, "-m", "swing.cli", "--config", str(cfg_path), "--data-dir", str(tmp_path),
                        "scan"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "market filter: PASS" in r.stdout
