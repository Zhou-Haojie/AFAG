"""
Generate AFAG objective-trace and adaptive-neighbour summaries.
"""
from __future__ import annotations
import os, sys, csv, json, time, argparse, importlib
import numpy as np

from algorithms_adaptive import AFAG
from algorithms import EPS
from data_loader import LOADERS

try:
    _runner = importlib.import_module("Running")
except ImportError:
    sys.exit("[ERROR] Running.py not found in this folder.")
FIXED, PREPROCESS_MODE = _runner.FIXED, _runner.PREPROCESS_MODE
preprocess, SEED = _runner.preprocess, _runner.SEED
DATA_ROOT, DATASETS = _runner.DATA_ROOT, _runner.DATASETS

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results_convergence")
TAB_DIR = os.path.join(HERE, "tables")
TOL = 1e-4



class AFAGTrace(AFAG):
    """AFAG that also records the neighbour count of every sample at every
    graph update.  k_i is read off the closed-form solution itself (the number
    of non-zero entries in row i of S^{x,new}), before damping, so it is
    exactly the quantity Eq. (ki) defines."""

    def fit(self, X):
        self.k_trace = []
        return super().fit(X)

    def _update_SX(self, *args, **kwargs):
        S = super()._update_SX(*args, **kwargs)
        self.k_trace.append((S > 0).sum(axis=1).astype(int))
        return S


def load_params(ds):
    p = os.path.join(HERE, "tuning_afag_wide", ds, "best_params.json")
    if not os.path.isfile(p):
        sys.exit(f"[ERROR] {p} not found -- these must be the parameters of the "
                 f"final experiment.")
    return json.load(open(p, encoding="utf-8"))["params"]


def stop_point(h):
    """First iteration at which the shipped rule would have stopped."""
    for t in range(1, len(h)):
        if h[t] <= h[t - 1] and abs(h[t] - h[t - 1]) / max(abs(h[t - 1]), EPS) < TOL:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=[])
    a = ap.parse_args()
    todo = a.datasets or DATASETS
    bad = [d for d in todo if d not in DATASETS]
    if bad:
        sys.exit(f"[ERROR] unknown dataset: {bad}")
    for d in (OUT_DIR, TAB_DIR):
        os.makedirs(d, exist_ok=True)

    conv_rows, ki_rows, traces = [], [], {}
    for ds in todo:
        X_raw, y, c = LOADERS[ds](DATA_ROOT)
        X = preprocess(X_raw, PREPROCESS_MODE[ds])
        n = X.shape[0]
        params = load_params(ds)
        kw = dict(FIXED["AFAG"]); kw.update(params); kw["tol"] = 0.0

        print(f"\n{'='*66}\n  {ds}   n={n}, d={X.shape[1]}, c={c}\n  {params}\n{'='*66}")
        t0 = time.time()
        m = AFAGTrace(c=c, **kw, seed=SEED).fit(X)
        h = np.asarray(m.obj_history, dtype=float)
        print(f"  {h.size} iterations in {time.time()-t0:.1f}s")

        d = np.diff(h)
        rel = d / np.maximum(np.abs(h[:-1]), EPS)
        rises = int(np.sum(rel > 1e-9))
        positive_rises = rel[rel > 1e-9]
        max_rise_pct = float(positive_rises.max() * 100) if positive_rises.size else 0.0
        drift = (h[-1] - h.min()) / max(abs(h.min()), EPS) * 100
        stop = stop_point(h)
        tail = h[len(h) // 2:]
        tail_range = (tail.max() - tail.min()) / max(abs(tail.min()), EPS) * 100
        conv_rows.append({
            "dataset": ds, "iterations": int(h.size),
            "stop_iter_tol1e-4": stop if stop is not None else "",
            "rises": rises, "steps": int(d.size),
            "max_rise_pct": round(max_rise_pct, 4),
            "J_start": float(h[0]), "J_min": float(h.min()), "J_end": float(h[-1]),
            "drift_above_min_pct": round(float(drift), 4),
            "second_half_range_pct": round(float(tail_range), 4)})
        print(f"  rises {rises}/{d.size}, max single rise {max_rise_pct:+.4f}%, "
              f"end {drift:+.4f}% above min, second-half range {tail_range:.4f}%")
        print(f"  shipped stopping rule would stop at iteration "
              f"{stop if stop is not None else 'none (runs to cap)'}")

        at = stop if stop is not None else len(m.k_trace) - 1
        k_used = m.k_trace[at]
        k_end = m.k_trace[-1]
        ki_rows.append({
            "dataset": ds, "n": n, "mu": params.get("mu"),
            "iter_used": at + 1,
            "k_mean": round(float(k_used.mean()), 2),
            "k_median": float(np.median(k_used)),
            "k_min": int(k_used.min()), "k_max": int(k_used.max()),
            "frac_k1": round(float(np.mean(k_used == 1)), 3),
            "n_distinct_k": int(np.unique(k_used).size),
            "k_mean_first_iter": round(float(m.k_trace[0].mean()), 2),
            "k_mean_iter_cap": round(float(k_end.mean()), 2)})
        print(f"  k_i at iteration {at + 1} (where the reported run stops): "
              f"mean {k_used.mean():.2f}, median {np.median(k_used):.0f}, "
              f"range [{k_used.min()}, {k_used.max()}], "
              f"{np.mean(k_used == 1)*100:.1f}% with k_i = 1, "
              f"{np.unique(k_used).size} distinct values")
        print(f"  (for reference: mean k_i {m.k_trace[0].mean():.2f} at the first "
              f"update, {k_end.mean():.2f} at iteration {len(m.k_trace)})")
        traces[ds] = {"objective": h.tolist(),
                      "k_mean": [float(k.mean()) for k in m.k_trace],
                      "k_used": k_used.tolist(), "stop": stop}

    with open(os.path.join(OUT_DIR, "convergence_summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(conv_rows[0])); w.writeheader()
        w.writerows(conv_rows)
    with open(os.path.join(OUT_DIR, "ki_summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ki_rows[0])); w.writeheader()
        w.writerows(ki_rows)
    json.dump(traces, open(os.path.join(OUT_DIR, "traces.json"), "w",
                           encoding="utf-8"))

    L = [r"% Generated by convergence_and_ki.py -- do not edit by hand.",
         r"\begin{table}[htbp]", r"\centering",
         r"\caption{Number of neighbours $k_i$ per sample in the AFAG graph.}",
         r"\label{tab:ki}",
         r"\begin{threeparttable}",
         r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Dataset & $n$ & $\mu$ & Mean & Median & Min & Max & Distinct \\",
         r"\midrule"]
    for r in ki_rows:
        mu = r["mu"]
        mu_s = (rf"$10^{{{int(round(np.log10(mu)))}}}$"
                if mu and abs(np.log10(mu) - round(np.log10(mu))) < 1e-9 else f"{mu:g}")
        L.append(rf"{r['dataset']} & {r['n']} & {mu_s} & {r['k_mean']:.2f} & "
                 rf"{r['k_median']:g} & {r['k_min']} & {r['k_max']} & "
                 rf"{r['n_distinct_k']} \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]\footnotesize\normalfont",
          r"\item Note: $k_i$ is counted in the graph of the last update before "
          r"the stopping rule of Algorithm~\ref{alg:afag} is met, at the "
          r"selected hyperparameters. Distinct: the number of distinct values "
          r"of $k_i$ over the $n$ samples.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}", ""]
    open(os.path.join(TAB_DIR, "table_ki.tex"), "w", encoding="utf-8").write("\n".join(L))

    print(f"\n{'='*66}\n  SUMMARY\n{'='*66}")
    print(f"  {'dataset':14s}{'stop@':>7}{'rises':>9}{'max rise':>11}"
          f"{'2nd-half':>10}   {'k range':>10}{'distinct':>9}{'k=1':>8}")
    for c_, k_ in zip(conv_rows, ki_rows):
        print(f"  {c_['dataset']:14s}{str(c_['stop_iter_tol1e-4'] or '-'):>7}"
              f"{str(c_['rises'])+'/'+str(c_['steps']):>9}"
              f"{c_['max_rise_pct']:+10.4f}%{c_['second_half_range_pct']:9.4f}%"
              f"   {str(k_['k_min'])+'-'+str(k_['k_max']):>10}"
              f"{k_['n_distinct_k']:9d}{k_['frac_k1']*100:7.1f}%")
    print(f"\n  [saved] {OUT_DIR}/, {TAB_DIR}/table_ki.tex\n")


if __name__ == "__main__":
    main()
