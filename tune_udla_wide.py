"""Evaluate the saved UDLA parameters or repeat the recorded wide-grid search.

Saved parameters and original search logs are read from tuning_udla_wide.
With --force, the wide-grid search starts from the (start) row in the saved
log and writes new tuning records to tuning_udla_rerun. Evaluation outputs
are saved to results_udla_rerun. The first-stage search is not reconstructed.

Usage:
    python tune_udla_wide.py --list
    python tune_udla_wide.py --all
    python tune_udla_wide.py GLIOMA
"""
from __future__ import annotations
import os, sys, csv, json, time, argparse, importlib
import numpy as np

from algorithms import evaluate_clustering
from algorithms_adaptive import UDLA
from data_loader import LOADERS

try:
    _runner = importlib.import_module("Running")
except ImportError:
    sys.exit("[ERROR] Running.py not found in this folder.")

FIXED, PREPROCESS_MODE = _runner.FIXED, _runner.PREPROCESS_MODE
DATASETS, preprocess, SEED = _runner.DATASETS, _runner.preprocess, _runner.SEED
DATA_ROOT, FEATURE_COUNTS = _runner.DATA_ROOT, _runner.FEATURE_COUNTS
N_RUNS = _runner.N_RUNS

HERE     = os.path.dirname(os.path.abspath(__file__))
TUNE_DIR = os.path.join(HERE, "tuning_udla_wide")
RERUN_TUNE_DIR = os.path.join(HERE, "tuning_udla_rerun")
RES_DIR  = os.path.join(HERE, "results_udla_rerun")

ROUNDS              = 2
TUNE_N_RUNS         = 5
TUNE_FEATURE_COUNTS = [40, 80, 120, 160, 200]

WIDE = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
SPACE = {"alpha": WIDE, "beta": WIDE, "gamma": WIDE, "lam": WIDE}
N_FITS = ROUNDS * sum(len(v) - 1 for v in SPACE.values()) + 1

ETA_MIN = {"Yale": 3.3, "ORL": 9.1, "warpAR10P": 18.7, "GLIOMA": 39.0,
           "TOX-171": 44.4, "CLL-SUB-111": 318.8}


def score(params, X, y, c):
    p = dict(FIXED["UDLA"]); p.update(params)
    try:
        m = UDLA(c=c, **p, seed=SEED).fit(X)
    except Exception as exc:
        return float("-inf"), {"error": f"{type(exc).__name__}: {exc}"[:160]}
    accs = []
    for l in TUNE_FEATURE_COUNTS:
        if l <= X.shape[1]:
            accs.append(evaluate_clustering(X[:, m.select(l)], y,
                                            n_runs=TUNE_N_RUNS,
                                            seed=SEED)["acc_mean"])
    if not accs:
        return float("-inf"), {"error": "no valid l"}
    return float(np.mean(accs)), {"error": ""}


def tune(ds, X, y, c, log, start_params):
    params = {key: float(start_params.get(key, 1.0)) for key in SPACE}
    best, st = score(params, X, y, c); used = 1
    start = best
    print(f"    start {params}  -> {best*100:.2f}%")
    log.append({"round": 0, "param": "(start)", "value": "", "score": best,
                **st, **params})
    for rnd in range(1, ROUNDS + 1):
        improved = False
        for pname, values in SPACE.items():
            cur = params[pname]
            for v in values:
                if np.isclose(v, cur):
                    continue
                trial = dict(params); trial[pname] = v
                s, st = score(trial, X, y, c); used += 1
                log.append({"round": rnd, "param": pname, "value": v,
                            "score": s, **st, **trial})
                if s > best:
                    best, params, improved = s, trial, True
            if not np.isclose(params[pname], cur):
                edge = " (EDGE)" if params[pname] in (WIDE[0], WIDE[-1]) else ""
                print(f"    r{rnd} {pname:14s} {cur} -> {params[pname]}{edge}"
                      f"   {best*100:.2f}%")
        if not improved:
            print(f"    r{rnd} no improvement, stopping")
            break
    return params, best, start, used


def evaluate(ds, X, y, c, params):
    p = dict(FIXED["UDLA"]); p.update(params)
    m = UDLA(c=c, **p, seed=SEED).fit(X)
    rows = []
    for l in [l for l in FEATURE_COUNTS if l <= X.shape[1]]:
        rows.append({"l": l, **evaluate_clustering(X[:, m.select(l)], y,
                                                   n_runs=N_RUNS, seed=SEED)})
    out = os.path.join(RES_DIR, ds); os.makedirs(out, exist_ok=True)
    tag = "UDLA (extended grid)"
    with open(os.path.join(out, "results.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "l", "ACC_mean%", "ACC_std%",
                    "NMI_mean%", "NMI_std%"])
        for e in rows:
            w.writerow([tag, e["l"], f"{e['acc_mean']*100:.3f}",
                        f"{e['acc_std']*100:.3f}", f"{e['nmi_mean']*100:.3f}",
                        f"{e['nmi_std']*100:.3f}"])
    with open(os.path.join(out, "raw_runs.json"), "w", encoding="utf-8") as f:
        json.dump({tag: [{"l": e["l"], "acc_runs": e["acc_runs"],
                          "nmi_runs": e["nmi_runs"]} for e in rows]}, f, indent=1)
    ia = int(np.argmax([e["acc_mean"] for e in rows]))
    inm = int(np.argmax([e["nmi_mean"] for e in rows]))
    return rows[ia]["acc_mean"] * 100, rows[ia]["l"], \
           rows[inm]["nmi_mean"] * 100, rows[inm]["l"]


def recorded_start(ds):
    path = os.path.join(TUNE_DIR, ds, "search_log.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Recorded search log not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("param") == "(start)" and row.get("round") == "0":
                try:
                    return {key: float(row[key]) for key in SPACE}
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid starting parameters in {path}") from exc
    raise ValueError(f"No (start) row found in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list or (not a.datasets and not a.all):
        print(f"\n  UDLA wide-grid search and evaluation")
        print(f"  grid for alpha, beta, gamma, lam: {WIDE}")
        print(f"  up to {N_FITS} candidate fits per dataset")
        tot = sum(ETA_MIN.values())
        for ds in sorted(DATASETS, key=lambda d: ETA_MIN[d]):
            print(f"    {ds:14s} ~{ETA_MIN[ds]:5.0f} min")
        print(f"    {'total':14s} ~{tot/60:5.1f} h  (+ full evaluation, a few min"
              f" per dataset)\n")
        print("  usage: python tune_udla_wide.py --all")
        print("  add --force to repeat the wide-grid search using the recorded start\n")
        return

    todo = DATASETS if a.all else a.datasets
    bad = [d for d in todo if d not in DATASETS]
    if bad:
        sys.exit(f"[ERROR] unknown dataset: {bad}")
    todo = sorted(todo, key=lambda d: ETA_MIN[d])

    summary, t0 = [], time.time()
    for ds in todo:
        bp_path = os.path.join(TUNE_DIR, ds, "best_params.json")
        print(f"{'='*68}\n  {ds}   (~{ETA_MIN[ds]:.0f} min)\n{'='*68}")
        X_raw, y, c = LOADERS[ds](DATA_ROOT)
        X = preprocess(X_raw, PREPROCESS_MODE[ds])

        if os.path.isfile(bp_path) and not a.force:
            rec = json.load(open(bp_path, encoding="utf-8"))
            params = rec["params"]
            print(f"  already tuned, reusing: {params}")
        elif a.force:
            t1, log = time.time(), []
            params, best, start, used = tune(ds, X, y, c, log,
                                             recorded_start(ds))
            rerun_dir = os.path.join(RERUN_TUNE_DIR, ds)
            os.makedirs(rerun_dir, exist_ok=True)
            rec = {"params": params, "score": best, "start_score": start,
                   "criterion": "mean_acc", "n_fits": used,
                   "seconds": time.time() - t1,
                   "protocol": {"rounds": ROUNDS, "tune_n_runs": TUNE_N_RUNS,
                                "tune_l": TUNE_FEATURE_COUNTS,
                                "grid": {k: list(map(float, v))
                                         for k, v in SPACE.items()},
                                "note": "UDLA wide-grid rerun from recorded start"}}
            with open(os.path.join(rerun_dir, "best_params.json"), "w",
                      encoding="utf-8") as f:
                json.dump(rec, f, indent=1)
            keys = sorted({k for r in log for k in r})
            with open(os.path.join(rerun_dir, "search_log.csv"), "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
                w.writerows(log)
            print(f"  search: {start*100:.2f}% -> {best*100:.2f}% "
                  f"({used} fits, {(time.time()-t1)/60:.1f} min)")
        else:
            raise FileNotFoundError(f"Saved parameters not found: {bp_path}")

        acc, la, nmi, ln = evaluate(ds, X, y, c, params)
        print(f"  full evaluation: ACC {acc:.2f}% @l={la}, NMI {nmi:.2f}% @l={ln}")
        summary.append((ds, acc, nmi))
        print()

    print(f"{'='*68}\n  SUMMARY   (total {(time.time()-t0)/60:.1f} min)\n{'='*68}")
    print(f"  {'dataset':14s}{'ACC':>10}{'NMI':>10}")
    for ds, acc, nmi in summary:
        print(f"  {ds:14s}{acc:10.2f}{nmi:10.2f}")
    if a.force:
        print(f"\n  [new tuning records] {RERUN_TUNE_DIR}/")
    print(f"  [evaluation results] {RES_DIR}/")


if __name__ == "__main__":
    main()
