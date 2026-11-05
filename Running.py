"""
Run the nine-method comparison on the six benchmark datasets.
"""
from __future__ import annotations
import os, sys, csv, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import (Baseline, VarianceScore, MCFS, JELSR, NDFS,
                        evaluate_clustering)
from algorithms_adaptive import AFAG, UDLA, CNAFS, SOGFS
from data_loader import LOADERS

HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(HERE, "dataset")
OUTPUT_ROOT = os.path.join(HERE, "results")

SEED     = 52
N_RUNS   = 20
MAX_ITER = 80
FEATURE_COUNTS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400, 500]
DATASETS = ["warpAR10P", "ORL", "Yale", "TOX-171", "CLL-SUB-111", "GLIOMA"]
PREPROCESS_MODE = {
    "warpAR10P": "row_norm", "ORL": "row_norm", "Yale": "row_norm",
    "TOX-171": "col_minmax", "CLL-SUB-111": "col_minmax",
    "GLIOMA": "col_minmax",
}

METHOD_ORDER = ["Baseline", "VarianceScore", "MCFS", "JELSR", "NDFS",
                "SOGFS", "CNAFS", "UDLA", "AFAG"]
NO_PARAMS = {"Baseline", "VarianceScore", "MCFS"}

SOURCE = {
    "AFAG":  ("tuning_afag_wide", None),
    "UDLA":  ("tuning_udla_wide", None),
    "JELSR": ("tuning_paper", "JELSR"),
    "NDFS":  ("tuning_paper", "NDFS"),
    "SOGFS": ("tuning_paper", "SOGFS"),
    "CNAFS": ("tuning_paper", "CNAFS"),
}
SEARCH_GRID = {
    "AFAG":  "1e-5 .. 1e5 (with UDLA, identical procedure)",
    "UDLA":  "1e-5 .. 1e5 (with AFAG, identical procedure)",
    "NDFS":  "{1e-6, 1e-4, ..., 1e6}, gamma = 1e8 fixed (Li et al., 2012)",
    "SOGFS": "{1e-3, ..., 1e3}, k = 5 (Nie et al., 2016)",
    "CNAFS": "{1e-3, ..., 1e3} for all five parameters (Yuan et al., 2022)",
    "JELSR": "{1e-6, 1e-4, ..., 1e6} (NDFS grid; original gives no range)",
    "MCFS":  "none -- LARS with cardinality constraint l (Cai et al., 2010)",
    "Baseline": "none", "VarianceScore": "none",
}

FIXED = {
    "MCFS":  {"k": 5, "sigma": 1e4},
    "JELSR": {"k": 5, "sigma": 1e4, "max_iter": MAX_ITER},
    "NDFS":  {"k": 5, "sigma": 1e4, "max_iter": MAX_ITER, "lam": 1e8},
    "SOGFS": {"k": 5, "max_iter": 20, "inner_iter": 1, "max_dim": 20000},
    "CNAFS": {"max_iter": 30, "max_dim": 20000},
    "UDLA":  {"k": 5, "sigma": 1e4, "max_iter": MAX_ITER},
    "AFAG":  {"k": 5, "sigma": 1e4, "max_iter": MAX_ITER},
}
CTOR = {"MCFS": MCFS, "JELSR": JELSR, "NDFS": NDFS, "SOGFS": SOGFS,
        "CNAFS": CNAFS, "UDLA": UDLA, "AFAG": AFAG}
NO_SEED = {"MCFS"}

OVERRIDES: dict = {}


def preprocess(X, mode):
    X = X.astype(np.float64)
    if mode == "row_norm":
        if X.min() < 0:
            X = X - X.min()
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n < 1e-10] = 1.0
        return X / n
    lo, hi = X.min(axis=0, keepdims=True), X.max(axis=0, keepdims=True)
    r = hi - lo; r[r < 1e-10] = 1.0
    return (X - lo) / r


def resolve(dataset):
    """(params, fixed, provenance) for every tuned method; stops if missing."""
    params, fixed, prov, missing = {}, {}, {}, []
    for m, (folder, key) in SOURCE.items():
        path = os.path.join(HERE, folder, dataset, "best_params.json")
        if not os.path.isfile(path):
            missing.append(f"{m}: {folder}/{dataset}/best_params.json"); continue
        rec = json.load(open(path, encoding="utf-8"))
        if key is not None:
            if key not in rec:
                missing.append(f"{m}: no entry in {folder}/{dataset}"); continue
            rec = rec[key]
        params[m] = dict(rec["params"])
        fixed[m] = dict(FIXED[m]); fixed[m].update(rec.get("fixed", {}))
        prov[m] = f"{folder}/{dataset}"
    if missing:
        sys.exit("[ERROR] tuning output missing -- not falling back to "
                 "defaults:\n  " + "\n  ".join(missing))
    for m, over in OVERRIDES.get(dataset, {}).items():
        params.setdefault(m, {}).update(over)
        prov[m] = prov.get(m, "") + "  + OVERRIDE " + json.dumps(over)
    return params, fixed, prov


def build(c, dataset):
    params, fixed, prov = resolve(dataset)
    out = {"Baseline": Baseline(), "VarianceScore": VarianceScore(),
           "MCFS": MCFS(c=c, **FIXED["MCFS"])}
    for m in SOURCE:
        kw = dict(fixed[m]); kw.update(params[m])
        if m not in NO_SEED:
            kw["seed"] = SEED
        out[m] = CTOR[m](c=c, **kw)
    return {m: out[m] for m in METHOD_ORDER}, params, fixed, prov


def run_dataset(name):
    print(f"\n{'='*66}\n  Dataset : {name}\n{'='*66}")
    X_raw, y, c = LOADERS[name](DATA_ROOT)
    X = preprocess(X_raw, PREPROCESS_MODE[name])
    n, d = X.shape
    counts = [l for l in FEATURE_COUNTS if l <= d]
    methods, params, fixed, prov = build(c, name)

    print(f"  n={n}, d={d}, c={c}, preprocess={PREPROCESS_MODE[name]}")
    for m in METHOD_ORDER:
        if m in params:
            print(f"    {m:<8} {params[m]}   [{prov[m]}]")
    print(f"  l grid: {counts}\n")

    results, unavailable, extra = {}, {}, {}
    for m, model in methods.items():
        print(f"  [{m}] fitting ... ", end="", flush=True)
        t0 = time.time()
        try:
            model.fit(X)
        except (MemoryError, np.linalg.LinAlgError, ValueError,
                FloatingPointError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"UNAVAILABLE\n      {reason[:150]}")
            unavailable[m] = reason; continue
        print(f"done ({time.time()-t0:.1f}s)")
        if m == "MCFS":
            extra["MCFS_n_ranked"] = int(getattr(model, "n_ranked", -1))
            if 0 <= extra["MCFS_n_ranked"] < max(counts):
                print(f"      note: LARS ranks {extra['MCFS_n_ranked']} features;"
                      f" for larger l the rest follow entry order")
        per_l = []
        for l in counts:
            r = evaluate_clustering(X[:, model.select(l=l)], y,
                                    n_runs=N_RUNS, seed=SEED)
            per_l.append({"l": l, **r})
            print(f"      l={l:3d}  ACC={r['acc_mean']*100:6.2f}±"
                  f"{r['acc_std']*100:4.2f}%   NMI={r['nmi_mean']*100:6.2f}±"
                  f"{r['nmi_std']*100:4.2f}%")
        results[m] = per_l
    return results, unavailable, params, fixed, prov, extra


def best(ents, key):
    i = int(np.argmax([e[f"{key}_mean"] for e in ents]))
    return ents[i][f"{key}_mean"] * 100, ents[i]["l"]


def consistency(name, results):
    """AFAG and UDLA were already evaluated by their tuning scripts."""
    notes = []
    for m, folder in (("AFAG", "results_afag_wide"), ("UDLA", "results_udla_wide")):
        p = os.path.join(HERE, folder, name, "results.csv")
        if not os.path.isfile(p) or m not in results:
            continue
        prev = {}
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            prev[int(r["l"])] = float(r["ACC_mean%"])
        diffs = [abs(e["acc_mean"] * 100 - prev[e["l"]])
                 for e in results[m] if e["l"] in prev]
        if diffs:
            worst = max(diffs)
            ok = worst < 0.01
            notes.append((m, worst, ok))
            print(f"  [check] {m} vs {folder}: max |dACC| = {worst:.4f}  "
                  f"{'OK' if ok else '<-- MISMATCH: code or data differ'}")
    return notes


def save(name, results, unavailable, params, fixed, prov, extra, checks):
    out = os.path.join(OUTPUT_ROOT, name); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "l", "ACC_mean%", "ACC_std%", "NMI_mean%",
                    "NMI_std%"])
        for m in METHOD_ORDER:
            for e in results.get(m, []):
                w.writerow([m, e["l"], f"{e['acc_mean']*100:.3f}",
                            f"{e['acc_std']*100:.3f}", f"{e['nmi_mean']*100:.3f}",
                            f"{e['nmi_std']*100:.3f}"])
    with open(os.path.join(out, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Best_ACC%", "Best_ACC_l", "Best_NMI%",
                    "Best_NMI_l", "Note"])
        for m in METHOD_ORDER:
            if m in unavailable:
                w.writerow([m, "", "", "", "", unavailable[m][:120]]); continue
            if m not in results:
                continue
            a, la = best(results[m], "acc"); nm, ln = best(results[m], "nmi")
            if m == "Baseline":
                la = ln = "all"
            w.writerow([m, f"{a:.3f}", la, f"{nm:.3f}", ln, ""])
    with open(os.path.join(out, "raw_runs.json"), "w", encoding="utf-8") as f:
        json.dump({m: [{"l": e["l"], "acc_runs": e["acc_runs"],
                        "nmi_runs": e["nmi_runs"]} for e in results[m]]
                   for m in METHOD_ORDER if m in results}, f, indent=1)
    with open(os.path.join(out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": name, "seed": SEED, "n_runs": N_RUNS,
                   "kmeans_init": "k-means++", "kmeans_n_init": 10,
                   "feature_counts": FEATURE_COUNTS,
                   "preprocess": PREPROCESS_MODE[name],
                   "params": params, "fixed": fixed, "provenance": prov,
                   "search_grid": SEARCH_GRID, "overrides":
                   OVERRIDES.get(name, {}), "unavailable": unavailable,
                   "consistency": [{"method": m, "max_abs_dacc": w_, "ok": ok}
                                   for m, w_, ok in checks], **extra},
                  f, indent=1)

    print(f"\n  {'='*62}\n  Summary : {name}\n  {'='*62}")
    print(f"  {'Method':<14}{'Best_ACC%':>11}{'@l':>6}{'Best_NMI%':>12}{'@l':>6}")
    print(f"  {'-'*49}")
    ua = None
    for m in METHOD_ORDER:
        if m in unavailable:
            print(f"  {m:<14}{'--':>11}{'':>6}{'--':>12}   "
                  f"({unavailable[m].split(':')[0]})"); continue
        if m not in results:
            continue
        a, la = best(results[m], "acc"); nm, ln = best(results[m], "nmi")
        if m == "Baseline":
            la = ln = "all"
        if m == "UDLA":
            ua = a
        tag = f"   vs UDLA {a-ua:+.2f}" if m == "AFAG" and ua is not None else ""
        print(f"  {m:<14}{a:>10.2f}%{str(la):>6}{nm:>11.2f}%{str(ln):>6}{tag}")
    print(f"\n  [saved] {out}/")


STYLE = {
    "Baseline":      dict(color="#7f7f7f", linestyle="--", marker="", lw=1.5),
    "VarianceScore": dict(color="#1f77b4", linestyle=":", marker="o", lw=1.4, markersize=5),
    "MCFS":          dict(color="#17becf", linestyle=":", marker="v", lw=1.4, markersize=5),
    "JELSR":         dict(color="#bcbd22", linestyle=":", marker="s", lw=1.4, markersize=5),
    "NDFS":          dict(color="#9467bd", linestyle=":", marker="D", lw=1.4, markersize=5),
    "SOGFS":         dict(color="#2ca02c", linestyle="-", marker="P", lw=2.0, markersize=7),
    "CNAFS":         dict(color="#8c564b", linestyle="-", marker="X", lw=2.0, markersize=7),
    "UDLA":          dict(color="#ff7f0e", linestyle="-", marker="^", lw=2.0, markersize=7),
    "AFAG":          dict(color="#d62728", linestyle="-", marker="*", lw=2.4, markersize=12),
}


def plot(name, results, metric):
    fig, ax = plt.subplots(figsize=(8.5, 5)); vals = []
    for m in METHOD_ORDER:
        if m not in results:
            continue
        ls = [e["l"] for e in results[m]]
        vs = [e[f"{metric}_mean"] * 100 for e in results[m]]; vals += vs
        if m == "Baseline":
            ax.axhline(vs[0], label=m, **STYLE[m])
        else:
            ax.plot(ls, vs, label=m, **STYLE[m])
    if vals:
        pad = max((max(vals) - min(vals)) * 0.1, 1.0)
        ax.set_ylim(min(vals) - pad, max(vals) + pad)
    ax.set_xlabel("Number of selected features"); ax.set_ylabel(f"{metric.upper()} (%)")
    ax.set_title(f"{name}  ({metric.upper()} vs. number of features)",
                 fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, name, f"comparison_{metric.upper()}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("--list", "-h", "--help"):
        print(f"\n  Final experiment -- {len(METHOD_ORDER)} methods x "
              f"{len(DATASETS)} datasets\n")
        for m in METHOD_ORDER:
            print(f"  {m:<14} {SEARCH_GRID[m]}")
        print()
        for ds in DATASETS:
            try:
                params, _, prov = resolve(ds)
            except SystemExit as e:
                print(f"  {ds}: {e}"); continue
            print(f"  {ds}")
            for m in ("AFAG", "UDLA", "JELSR", "NDFS", "SOGFS", "CNAFS"):
                print(f"    {m:<6} {params[m]}")
        print("\n  usage: python Runing_final.py [dataset | --all]\n")
        return

    todo = DATASETS if args[0] == "--all" else args
    bad = [a for a in todo if a not in DATASETS]
    if bad:
        sys.exit(f"[ERROR] unknown dataset: {bad}")
    for ds in todo:
        resolve(ds)

    t0 = time.time()
    for name in todo:
        t1 = time.time()
        try:
            res, un, params, fixed, prov, extra = run_dataset(name)
            checks = consistency(name, res)
            save(name, res, un, params, fixed, prov, extra, checks)
            plot(name, res, "acc"); plot(name, res, "nmi")
            print(f"\n  {name} elapsed: {(time.time()-t1)/60:.1f} min")
        except Exception as exc:
            print(f"\n  [ERROR] {name}: {exc}")
            import traceback; traceback.print_exc()
    print(f"\n{'='*66}\n  All done in {(time.time()-t0)/60:.1f} min.  "
          f"Results in {OUTPUT_ROOT}\n{'='*66}\n")


if __name__ == "__main__":
    main()
