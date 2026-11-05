"""Tune comparison methods on their stated parameter grids.

Protocol
--------
JELSR, NDFS, SOGFS, and CNAFS are tuned on the grids listed below.

Ranges, with provenance (cite these in Section 4.4)
---------------------------------------------------
  NDFS   alpha, beta in {1e-6, 1e-4, ..., 1e6}; gamma (orthogonality) fixed at
         1e8; k = 5.  Li et al., AAAI 2012, "Experimental Settings".
  SOGFS  gamma in {1e-3, 1e-2, ..., 1e3}; k = 5 fixed.  Nie, Zhu & Li,
         AAAI 2016 and TKDE 2021, Section 5.2.
  CNAFS  alpha, beta, lambda, gamma, epsilon in {1e-3, ..., 1e3}.  Yuan et al.,
         IEEE T-Cybernetics 2022 (range as read from the paper by the author).
  JELSR  the original gives no range ("heuristic grid search"); searched on the
         NDFS grid {1e-6, 1e-4, ..., 1e6}; k = 5.

Search procedure
----------------
Coordinate descent with two sweeps; mean ACC over l in {40, 80, 120, 160, 200}
and five k-means runs. Existing saved parameters are used as warm starts.

"""
from __future__ import annotations
import os, sys, csv, json, time, argparse, importlib
import numpy as np

from algorithms import JELSR, NDFS, evaluate_clustering
from algorithms_adaptive import SOGFS, CNAFS
from data_loader import LOADERS

try:
    _runner = importlib.import_module("Runing_final")
except ImportError:
    sys.exit("[ERROR] Runing_final.py not found in this folder.")

PREPROCESS_MODE, DATASETS = _runner.PREPROCESS_MODE, _runner.DATASETS
preprocess, SEED, DATA_ROOT = _runner.preprocess, _runner.SEED, _runner.DATA_ROOT

HERE     = os.path.dirname(os.path.abspath(__file__))
TUNE_DIR = os.path.join(HERE, "tuning_paper")

ROUNDS              = 2
TUNE_N_RUNS         = 5
TUNE_FEATURE_COUNTS = [40, 80, 120, 160, 200]

NDFS_GRID  = [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6]
THREE_GRID = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]

SPACE = {
    "JELSR": {"alpha": NDFS_GRID, "beta": NDFS_GRID},
    "NDFS":  {"alpha": NDFS_GRID, "beta": NDFS_GRID},
    "SOGFS": {"gamma": THREE_GRID},
    "CNAFS": {"alpha": THREE_GRID, "beta": THREE_GRID, "lam": THREE_GRID,
              "gamma": THREE_GRID, "epsilon": THREE_GRID},
}
FIXED = {
    "JELSR": dict(_runner.FIXED["JELSR"]),
    "NDFS":  dict(_runner.FIXED["NDFS"]),
    "SOGFS": dict(_runner.FIXED["SOGFS"]),
    "CNAFS": dict(_runner.FIXED["CNAFS"]),
}
SOURCE = {
    "JELSR": "NDFS grid (original JELSR specifies no range)",
    "NDFS":  "Li et al., AAAI 2012",
    "SOGFS": "Nie, Zhu & Li, AAAI 2016 / TKDE 2021",
    "CNAFS": "Yuan et al., IEEE T-Cybernetics 2022",
}
CTOR = {"JELSR": JELSR, "NDFS": NDFS, "SOGFS": SOGFS,
        "CNAFS": CNAFS}
METHODS = list(SPACE)
NO_SEED = set()

# rough minutes per (method, dataset), from the symmetric run scaled by fits
ETA = {"JELSR": dict(warpAR10P=5, ORL=3, Yale=2, **{"TOX-171": 27,
                     "CLL-SUB-111": 94}, GLIOMA=16),
       "NDFS":  dict(warpAR10P=9, ORL=3, Yale=2, **{"TOX-171": 41,
                     "CLL-SUB-111": 238}, GLIOMA=26),
       "SOGFS": dict(warpAR10P=2, ORL=1, Yale=1, **{"TOX-171": 55,
                     "CLL-SUB-111": 345}, GLIOMA=22),
       "CNAFS": dict(warpAR10P=3, ORL=4, Yale=2, **{"TOX-171": 45,
                     "CLL-SUB-111": 165}, GLIOMA=18)}


def n_fits(m):
    return ROUNDS * sum(len(v) - 1 for v in SPACE[m].values()) + 1


def snap(v, grid):
    """Nearest grid point to v on a log scale."""
    lg = np.log10(np.asarray(grid, dtype=float))
    return float(grid[int(np.argmin(np.abs(lg - np.log10(max(float(v), 1e-30)))))])


def warm_start(m, ds):
    path = os.path.join(TUNE_DIR, ds, "best_params.json")
    records = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}
    prev = records.get(m, {}).get("params", {})
    return {p: snap(prev.get(p, 1.0), g) for p, g in SPACE[m].items()}


def score(m, params, X, y, c):
    kw = dict(FIXED[m]); kw.update(params)
    if m not in NO_SEED:
        kw["seed"] = SEED
    try:
        model = CTOR[m](c=c, **kw).fit(X)
    except Exception as exc:
        return float("-inf"), f"{type(exc).__name__}: {exc}"[:160]
    accs = [evaluate_clustering(X[:, model.select(l)], y, n_runs=TUNE_N_RUNS,
                                seed=SEED)["acc_mean"]
            for l in TUNE_FEATURE_COUNTS if l <= X.shape[1]]
    return (float(np.mean(accs)), "") if accs else (float("-inf"), "no valid l")


def tune(m, ds, X, y, c, log):
    params = warm_start(m, ds)
    best, err = score(m, params, X, y, c); used = 1
    print(f"    start {params} -> {best*100:.2f}%" + (f"  [{err}]" if err else ""))
    log.append({"method": m, "round": 0, "param": "(start)", "value": "",
                "score": best, "error": err, **params})
    for rnd in range(1, ROUNDS + 1):
        improved = False
        for p, grid in SPACE[m].items():
            cur = params[p]
            for v in grid:
                if np.isclose(v, cur):
                    continue
                trial = dict(params); trial[p] = v
                s, err = score(m, trial, X, y, c); used += 1
                log.append({"method": m, "round": rnd, "param": p, "value": v,
                            "score": s, "error": err, **trial})
                if s > best:
                    best, params, improved = s, trial, True
            if not np.isclose(params[p], cur):
                edge = " (EDGE)" if params[p] in (grid[0], grid[-1]) else ""
                print(f"    r{rnd} {p:10s} {cur} -> {params[p]}{edge}   "
                      f"{best*100:.2f}%")
        if not improved:
            print(f"    r{rnd} no improvement, stopping"); break
    return params, best, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list or (not a.datasets and not a.all):
        print("\n  comparison methods on their papers' recommended ranges\n")
        for m in METHODS:
            print(f"  {m:6s} {n_fits(m):3d} fits   {SOURCE[m]}")
            for p, g in SPACE[m].items():
                print(f"         {p:8s} {g}")
            fx = {k: v for k, v in FIXED[m].items() if k in ("k", "lam")}
            if fx:
                print(f"         fixed    {fx}")
        tot = sum(ETA[m][d] for m in METHODS for d in DATASETS)
        print(f"\n  estimated total ~{tot/60:.1f} h "
              f"(CLL-SUB-111 alone ~{sum(ETA[m]['CLL-SUB-111'] for m in METHODS)/60:.1f} h)")
        print("\n  usage: python tune_paper_ranges.py --all\n")
        return

    todo = DATASETS if a.all else a.datasets
    bad = [d for d in todo if d not in DATASETS]
    if bad:
        sys.exit(f"[ERROR] unknown dataset: {bad}")
    todo = sorted(todo, key=lambda d: sum(ETA[m][d] for m in a.methods))

    t0 = time.time()
    for ds in todo:
        out = os.path.join(TUNE_DIR, ds); os.makedirs(out, exist_ok=True)
        bp = os.path.join(out, "best_params.json")
        rec = json.load(open(bp, encoding="utf-8")) if os.path.isfile(bp) else {}

        print(f"\n{'='*66}\n  {ds}\n{'='*66}")
        X_raw, y, c = LOADERS[ds](DATA_ROOT)
        X = preprocess(X_raw, PREPROCESS_MODE[ds])

        log = []
        for m in sorted(a.methods, key=lambda m: ETA[m][ds]):
            if m in rec and not a.force:
                print(f"  [{m}] already tuned: {rec[m]['params']}"); continue
            print(f"  [{m}] ~{ETA[m][ds]} min, {n_fits(m)} fits, {SOURCE[m]}")
            t1 = time.time()
            params, best, used = tune(m, ds, X, y, c, log)
            rec[m] = {"params": params, "score": best, "criterion": "mean_acc",
                      "n_fits": used, "seconds": time.time() - t1,
                      "fixed": {k: v for k, v in FIXED[m].items()},
                      "source": SOURCE[m],
                      "protocol": {"rounds": ROUNDS, "tune_n_runs": TUNE_N_RUNS,
                                   "tune_l": TUNE_FEATURE_COUNTS,
                                   "grid": {k: list(map(float, v))
                                            for k, v in SPACE[m].items()}}}
            json.dump(rec, open(bp, "w", encoding="utf-8"), indent=1)
            print(f"    -> {params}   {best*100:.2f}%  "
                  f"({used} fits, {(time.time()-t1)/60:.1f} min)\n")

        if log:
            path = os.path.join(out, "search_log.csv")
            keys = sorted({k for r in log for k in r})
            new = not os.path.isfile(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                if new:
                    w.writeheader()
                w.writerows(log)

    print(f"\n{'='*66}\n  done in {(time.time()-t0)/60:.1f} min\n"
          f"  [saved] {TUNE_DIR}/\n{'='*66}\n")


if __name__ == "__main__":
    main()
