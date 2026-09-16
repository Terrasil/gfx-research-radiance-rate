#!/usr/bin/env python3
from __future__ import annotations
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

def num(row, key):
    try:
        return float(row[key])
    except Exception:
        return math.nan

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/publication")
    src = root / "raw.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
    print(f"rows: {len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    print("method,gpu_case_median,relative_rmse_median,violation_rate_median,rate4_median")
    for method in sorted(groups):
        g = groups[method]
        def med(key):
            v = [num(r, key) for r in g if math.isfinite(num(r, key))]
            return statistics.median(v) if v else math.nan
        print(f"{method},{med('gpu_median_ms'):.6f},{med('relative_rmse'):.8f},"
              f"{med('violation_rate'):.6f},{med('rate4'):.6f}")
    for row in rows:
        total = num(row, "rate1") + num(row, "rate2") + num(row, "rate4")
        if not (0.999 <= total <= 1.001):
            raise SystemExit(f"invalid shading-rate fractions in row: {row}")
    print("rate-fraction contract: PASS")
if __name__ == "__main__":
    main()
