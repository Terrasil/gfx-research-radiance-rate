#!/usr/bin/env python3
"""Audit raw queries and compare the shading-rate methods within each workload."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

FULL = "full-rate-direct"
SERIAL = "feature-budget"
PARALLEL = "parallel-feature"
EXPECTED = {"full-rate","fixed-2x2","fixed-4x4","heuristic","radiance-budget",SERIAL,PARALLEL,FULL}
ERROR_LIMITS = (0.0025, 0.005, 0.01, 0.02)


def read(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    manifest = dict(line.split("=",1) for line in (root/"manifest.txt").read_text().splitlines())
    if manifest.get("completed") != "1":
        raise ValueError("The run is incomplete")
    planned = int(manifest["planned"])
    samples = set(range(int(manifest["samples"])))
    times = defaultdict(dict)
    for row in read(root/"timings.csv"):
        case = row["case"],row["block"],row["method"]
        i, ns = int(row["sample"]), int(row["gpu_ns"])
        if ns<=0 or i in times[case]:
            raise ValueError("Invalid or duplicated timestamp")
        times[case][i] = ns/1e6
    quality = read(root/"raw.csv")
    if len(quality)!=planned or len(times)!=planned:
        raise ValueError("Timing or quality rows are incomplete")

    fields = ("width","height","lights","camera","roughness_scale")
    blocks = defaultdict(dict)
    cases = set()
    scores = ("rmse","gradient_rmse","relative_rmse","violation_rate","rate1","rate2","rate4")
    for row in quality:
        key = row["case"],row["block"],row["method"]
        if key in cases or samples != set(times[key]):
            raise ValueError("Duplicate case or incomplete timestamp series")
        cases.add(key)
        for name in scores:
            if not math.isfinite(float(row[name])):
                raise ValueError(f"Invalid {name}")
        if abs(1-sum(float(row[k]) for k in ("rate1","rate2","rate4"))) > 0.001:
            raise ValueError("Invalid shading rates")
        row["ms"] = median(times[key].values())
        if not math.isclose(row["ms"],float(row["gpu_median_ms"]),rel_tol=1e-5):
            raise ValueError("The timing summary disagrees with raw timestamps")
        case = tuple(row[k] for k in fields)
        blocks[case,row["error_budget"],int(row["block"])][row["method"]] = row

    paired = defaultdict(list)
    for (work,budget,block),methods in blocks.items():
        if set(methods) != EXPECTED:
            raise ValueError(f"Incomplete methods for {work}/{budget}/{block}")
        if any(float(methods[k]["rmse"]) > 1e-6 for k in (FULL,"full-rate")):
            raise ValueError("A full-rate pass disagrees with the reference")
        for name in scores:
            if abs(float(methods[SERIAL][name])-float(methods[PARALLEL][name])) > 1e-6:
                raise ValueError(f"The parallel reduction changed the serial gate output ({name})")
        for method,row in methods.items():
            paired[work,budget,method].append(row | {
                "delta_ms":row["ms"]-methods[FULL]["ms"],
                "delta_serial":row["ms"]-methods[SERIAL]["ms"],
            })

    arms = []
    for (work,budget,method),group in sorted(paired.items()):
        row = dict(zip(fields,work))
        measured = median(r["ms"] for r in group)
        arms.append(row | dict(budget=budget,method=method,blocks=len(group),median_ms=measured,
                               delta_ms_direct=median(r["delta_ms"] for r in group),
                               delta_ms_serial=median(r["delta_serial"] for r in group),
                               relative_rmse=median(float(r["relative_rmse"]) for r in group),
                               gradient_rmse=median(float(r["gradient_rmse"]) for r in group),
                               false_safe_rate=median(float(r["false_safe_rate"]) for r in group) if method=="radiance-budget" else None,
                               predictor_tiles=group[0]["predictor_tiles"] if method=="radiance-budget" else None,
                               violation_rate=median(float(r["violation_rate"]) for r in group)))
    write(root/"quality_cost.csv",arms)
    by_work = defaultdict(list)
    for row in arms:
        by_work[tuple(row[k] for k in fields)].append(row)
    matched = []
    for work,candidates in sorted(by_work.items()):
        for limit in ERROR_LIMITS:
            allowed = [r for r in candidates if r["relative_rmse"] <= limit]
            best = min(allowed,key=lambda r:(r["median_ms"],r["relative_rmse"]))
            matched.append(dict(zip(fields,work)) | dict(relative_rmse_limit=limit,method=best["method"],
                                budget=best["budget"],median_ms=best["median_ms"],matched_delta_ms=best["delta_ms_direct"],
                                relative_rmse=best["relative_rmse"]))
    write(root/"fixed_quality.csv",matched)
    print(f"{len(quality)} paired lighting cases. Full-rate and serial/parallel equivalence checks passed.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path,nargs="?",default=Path("results/publication"))
    analyze(p.parse_args().results)
