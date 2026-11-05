"""Run AFAG ablation and parameter-sensitivity experiments.
Ablations:
- AFAG-noGraph: keeps the initial sample graph fixed.
- AFAG-noFused: updates the graph using rescaled raw distances.
- AFAG-noDamp: disables graph-update damping.
"""
from __future__ import annotations
import os, sys, csv, json, time, argparse, importlib
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import evaluate_clustering, _sq_dists, _knn_graph, _split
from algorithms_adaptive import AFAG
from data_loader import LOADERS

try:
    _runner = importlib.import_module("Running")
except ImportError:
    sys.exit("[ERROR] Running.py not found in this folder.")
FIXED = _runner.FIXED
preprocess, SEED = _runner.preprocess, _runner.SEED
DATA_ROOT, DATASETS = _runner.DATA_ROOT, _runner.DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))
L_MAIN = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
MU_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
RHO_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
EPS = 1e-10
RED = "#E60012"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- variants
def closed_form_S(H, mu):
    """Row-wise closed form of the graph sub-problem; identical to AFAG's."""
    n = H.shape[0]
    H_ns = H.copy(); np.fill_diagonal(H_ns, np.inf)
    sH = np.sort(H_ns, axis=1)[:, :n - 1]
    cumH = np.cumsum(sH, axis=1)
    ks = np.arange(1, n).reshape(1, -1)
    nus = (2.0 * mu + cumH) / ks
    valid = sH < nus
    k_row = np.where(valid.any(1), (valid * ks).max(1), 1).astype(int)
    nu_row = (2.0 * mu + cumH[np.arange(n), k_row - 1]) / k_row
    S = np.maximum((nu_row[:, None] - H) / (2.0 * mu), 0.0)
    np.fill_diagonal(S, 0.0)
    return S


class AFAGTrace(AFAG):
    """Records the neighbour count of every sample at every graph update."""
    def fit(self, X):
        self.k_trace = []
        return super().fit(X)

    def _update_SX(self, *a, **kw):
        S = super()._update_SX(*a, **kw)
        self.k_trace.append((S > 0).sum(1))
        return S


class AFAGNoGraph(AFAG):
    """The warm-start graph is kept for the whole run."""
    def fit(self, X):
        S0 = _knn_graph(np.asarray(X, float), self.k, self.sigma, "sample")
        self._S0, _ = _split(S0)
        return super().fit(X)

    def _update_SX(self, R, Y, beta, gamma, mu):
        return self._S0.copy()


class AFAGNoFused(AFAG):
    """Graph updated from raw distances, rescaled to the fused-distance mean."""
    def fit(self, X):
        Xf = np.asarray(X, float)
        self._Draw = _sq_dists(Xf)
        return super().fit(X)

    def _update_SX(self, R, Y, beta, gamma, mu):
        Hf = (beta / 2.0) * _sq_dists(R) + (gamma / 2.0) * _sq_dists(Y)
        off = ~np.eye(Hf.shape[0], dtype=bool)
        scale = Hf[off].mean() / max(self._Draw[off].mean(), EPS)
        return closed_form_S(self._Draw * scale, mu)


VARIANTS = {"AFAG-noGraph": AFAGNoGraph, "AFAG-noFused": AFAGNoFused,
            "AFAG-noDamp": AFAG, "AFAG": AFAG}


# ---------------------------------------------------------------- helpers
def load(ds):
    X_raw, y, c = LOADERS[ds](DATA_ROOT)
    X = preprocess(X_raw, PREPROCESS_MODE[ds])
    p = os.path.join(HERE, "tuning_afag_wide", ds, "best_params.json")
    if not os.path.isfile(p):
        sys.exit(f"[ERROR] {p} not found.")
    return X, y, c, json.load(open(p, encoding="utf-8"))["params"]


def evaluate(model, X, y, n_runs):
    """Best mean over l <= 200, with the paired runs at that l."""
    out = []
    for l in [l for l in L_MAIN if l <= X.shape[1]]:
        r = evaluate_clustering(X[:, model.select(l)], y, n_runs=n_runs, seed=SEED)
        out.append({"l": l, **r})
    ba = max(out, key=lambda e: e["acc_mean"])
    bn = max(out, key=lambda e: e["nmi_mean"])
    return ba, bn


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


# ---------------------------------------------------------------- ablation
def run_ablation(todo, n_runs):
    os.makedirs(os.path.join(HERE, "results_ablation"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    rows, raw = [], {}
    for ds in todo:
        X, y, c, prm = load(ds)
        print(f"\n{'='*66}\n  ablation: {ds}   (selected rho = {prm['graph_damping']})\n{'='*66}")
        res = {}
        for name, cls in VARIANTS.items():
            kw = dict(FIXED["AFAG"]); kw.update(prm)
            if name == "AFAG-noDamp":
                kw["graph_damping"] = 1.0
            t0 = time.time()
            m = cls(c=c, **kw, seed=SEED).fit(X)
            ba, bn = evaluate(m, X, y, n_runs)
            res[name] = (ba, bn)
            print(f"  {name:14s} ACC {ba['acc_mean']*100:6.2f}±{ba['acc_std']*100:4.2f} "
                  f"@{ba['l']:<4} NMI {bn['nmi_mean']*100:6.2f}±{bn['nmi_std']*100:4.2f} "
                  f"@{bn['l']:<4} ({time.time()-t0:.0f}s)")
        full_a, full_n = res["AFAG"]
        pa, pn = [], []
        for name in list(VARIANTS)[:3]:
            a, b = np.array(full_a["acc_runs"]), np.array(res[name][0]["acc_runs"])
            pa.append(1.0 if np.allclose(a, b) else float(stats.wilcoxon(a, b).pvalue))
            a, b = np.array(full_n["nmi_runs"]), np.array(res[name][1]["nmi_runs"])
            pn.append(1.0 if np.allclose(a, b) else float(stats.wilcoxon(a, b).pvalue))
        ha, hn = holm(pa), holm(pn)
        for k, name in enumerate(VARIANTS):
            ba, bn = res[name]
            rows.append({"dataset": ds, "variant": name, "rho_selected": prm["graph_damping"],
                         "acc": round(ba["acc_mean"] * 100, 3),
                         "acc_std": round(ba["acc_std"] * 100, 3), "acc_l": ba["l"],
                         "nmi": round(bn["nmi_mean"] * 100, 3),
                         "nmi_std": round(bn["nmi_std"] * 100, 3), "nmi_l": bn["l"],
                         "p_holm_acc_vs_AFAG": "" if name == "AFAG" else round(float(ha[k]), 5),
                         "p_holm_nmi_vs_AFAG": "" if name == "AFAG" else round(float(hn[k]), 5)})
        raw[ds] = {n: {"acc_runs": res[n][0]["acc_runs"], "nmi_runs": res[n][1]["nmi_runs"]}
                   for n in VARIANTS}

    with open(os.path.join(HERE, "results_ablation", "ablation.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(raw, open(os.path.join(HERE, "results_ablation", "ablation_raw.json"), "w"))

    # LaTeX: variants as rows, datasets as columns, ACC block then NMI block
    L = [r"% Generated by ablation_sensitivity.py -- do not edit by hand.",
         r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Ablation study of AFAG.}",
         r"\label{tab:ablation}",
         r"\begin{threeparttable}",
         r"\begin{tabular}{ll" + "c" * len(todo) + "}", r"\hline",
         r"Metric & Variant & " + " & ".join(todo) + r" \\", r"\hline"]
    for metric, key in (("ACC", "acc"), ("NMI", "nmi")):
        for k, name in enumerate(VARIANTS):
            cells = []
            for ds in todo:
                r = next(x for x in rows if x["dataset"] == ds and x["variant"] == name)
                full = next(x for x in rows if x["dataset"] == ds and x["variant"] == "AFAG")
                t = rf"{r[key]:.2f}$\pm${r[key+'_std']:.2f}"
                p = r[f"p_holm_{key}_vs_AFAG"]
                if p != "" and p < 0.05:
                    t += r"$^{\downarrow}$" if r[key] < full[key] else r"$^{\uparrow}$"
                # bold the best value in the column, whichever variant it is
                best = max(x[key] for x in rows if x["dataset"] == ds)
                if np.isclose(r[key], best):
                    t = rf"\textbf{{{t}}}"
                cells.append(t)
            L.append((metric if k == 0 else "") + f" & {name} & " + " & ".join(cells) + r" \\")
        L.append(r"\hline")
    L.append(r"Selected $\rho$ & & " + " & ".join(
        f"{next(x for x in rows if x['dataset']==ds)['rho_selected']:g}" for ds in todo) + r" \\")
    L += [r"\hline", r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]\footnotesize\normalfont",
          r"\item Note: Best mean (\%) $\pm$ standard deviation over "
          r"$l\in\{20,\ldots,200\}$. $^{\downarrow}$ and $^{\uparrow}$ mark a variant "
          r"that is significantly worse or better than the complete AFAG (Wilcoxon "
          r"signed-rank test, Holm-corrected, $\alpha=0.05$). Bold marks the best "
          r"value in each column.",
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}", ""]
    # booktabs rules: first \hline -> \toprule, last -> \bottomrule, others -> \midrule
    _h = [i for i, l in enumerate(L) if l.strip() == r"\hline"]
    for _k, _i in enumerate(_h):
        L[_i] = (r"\toprule" if _k == 0 else
                 r"\bottomrule" if _k == len(_h) - 1 else r"\midrule")
    open(os.path.join(HERE, "tables", "table_ablation.tex"), "w", encoding="utf-8").write("\n".join(L))

    print(f"\n{'='*66}\n  ABLATION SUMMARY (ACC, change vs AFAG; * = significant)\n{'='*66}")
    print(f"  {'dataset':14s}" + "".join(f"{v:>15s}" for v in list(VARIANTS)[:3]))
    for ds in todo:
        full = next(x for x in rows if x["dataset"] == ds and x["variant"] == "AFAG")
        cells = []
        for v in list(VARIANTS)[:3]:
            r = next(x for x in rows if x["dataset"] == ds and x["variant"] == v)
            star = "*" if r["p_holm_acc_vs_AFAG"] < 0.05 else " "
            cells.append(f"{r['acc']-full['acc']:+13.2f}{star} ")
        print(f"  {ds:14s}" + "".join(cells))


# ---------------------------------------------------------------- sensitivity
def run_sensitivity(todo, n_runs, fmt):
    os.makedirs(os.path.join(HERE, "results_sensitivity"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    rows = []
    for ds in todo:
        X, y, c, prm = load(ds)
        print(f"\n{'='*66}\n  sensitivity: {ds}   selected mu={prm['mu']}, rho={prm['graph_damping']}\n{'='*66}")
        for pname, grid in (("mu", MU_GRID), ("graph_damping", RHO_GRID)):
            for v in grid:
                kw = dict(FIXED["AFAG"]); kw.update(prm); kw[pname] = v
                m = AFAGTrace(c=c, **kw, seed=SEED).fit(X)
                ba, bn = evaluate(m, X, y, n_runs)
                kmean = float(m.k_trace[-1].mean()) if m.k_trace else float("nan")
                rows.append({"dataset": ds, "param": pname, "value": v,
                             "selected": bool(np.isclose(v, prm[pname])),
                             "acc": round(ba["acc_mean"] * 100, 3),
                             "nmi": round(bn["nmi_mean"] * 100, 3),
                             "k_mean": round(kmean, 2)})
                print(f"  {pname:13s} {v:<8g} ACC {ba['acc_mean']*100:6.2f}  "
                      f"NMI {bn['nmi_mean']*100:6.2f}  k_mean {kmean:7.2f}"
                      + ("   <- selected" if np.isclose(v, prm[pname]) else ""))
    with open(os.path.join(HERE, "results_sensitivity", "sensitivity.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    for pname, label, fname in (("mu", r"$\mu$", "mu"), ("graph_damping", r"$\rho$", "rho")):
        nr = (len(todo) + 2) // 3
        fig, axes = plt.subplots(nr, 3, figsize=(7.1, 2.3 * nr))
        flat = np.ravel(axes)
        for k, (ax, ds) in enumerate(zip(flat, todo)):
            r = [x for x in rows if x["dataset"] == ds and x["param"] == pname]
            xs = [x["value"] for x in r]
            ax.plot(xs, [x["acc"] for x in r], color=RED, marker="o", ms=3.5,
                    lw=1.6, label="ACC")
            ax.plot(xs, [x["nmi"] for x in r], color="#0072B2", marker="s",
                    ms=3.2, lw=1.2, ls="--", label="NMI")
            sel = [x for x in r if x["selected"]]
            for s in sel:
                ax.plot(s["value"], s["acc"], marker="o", ms=8, mfc="none",
                        mec="black", mew=1.0, zorder=5)
            if pname == "mu":
                ax.set_xscale("log")
                ax2 = ax.twinx()
                ax2.plot(xs, [x["k_mean"] for x in r], color="#888888", lw=0.9,
                         ls=":", marker="^", ms=2.5, label=r"mean $k_i$")
                ax2.set_yscale("log")
                ax2.spines["right"].set_visible(True)
                ax2.tick_params(axis="y", labelsize=7, colors="#666666")
                if k % 3 == 2:
                    ax2.set_ylabel(r"mean $k_i$", color="#666666", fontsize=8)
            ax.set_title(f"({chr(97 + k)}) {ds}", loc="left", fontweight="bold", pad=3)
            ax.grid(True, axis="y", color="#DDDDDD", lw=0.6)
            if k % 3 == 0:
                ax.set_ylabel("(%)")
            if k >= len(todo) - 3:
                ax.set_xlabel(label)
        for ax in flat[len(todo):]:
            ax.axis("off")
        h, l = flat[0].get_legend_handles_labels()
        if pname == "mu":
            h.append(plt.Line2D([], [], color="#888888", ls=":", marker="^", ms=3))
            l.append(r"mean $k_i$ (right axis)")
        h.append(plt.Line2D([], [], marker="o", ms=7, mfc="none", mec="black", ls=""))
        l.append("selected value")
        fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0, w_pad=1.0)
        fig.savefig(os.path.join(HERE, "figures", f"fig_sensitivity_{fname}.{fmt}"),
                    bbox_inches="tight")
        plt.close(fig)
    print(f"\n  [saved] results_sensitivity/sensitivity.csv, "
          f"figures/fig_sensitivity_mu.{fmt}, figures/fig_sensitivity_rho.{fmt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["ablation", "sensitivity", "all"])
    ap.add_argument("datasets", nargs="*", default=[])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    a = ap.parse_args()
    todo = a.datasets or DATASETS
    bad = [d for d in todo if d not in DATASETS]
    if bad:
        sys.exit(f"[ERROR] unknown dataset: {bad}")
    t0 = time.time()
    if a.part in ("ablation", "all"):
        run_ablation(todo, a.runs)
    if a.part in ("sensitivity", "all"):
        run_sensitivity(todo, a.runs, a.format)
    print(f"\n  total {(time.time()-t0)/60:.1f} min\n")


if __name__ == "__main__":
    main()
