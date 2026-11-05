# Seven unsupervised feature selection methods:
# Baseline / VarianceScore / MCFS / JELSR / NDFS / UDLA / AFAG
# Common interface: model.fit(X) then model.select(l) -> indices of top-l features.

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score

EPS      = 1e-10
CLIP_MAX = 1e6

def _sq_dists(Z):  # pairwise squared Euclidean distances between rows of Z
    return squareform(pdist(Z, "sqeuclidean"))

def _knn_graph(X, k=5, sigma=1e4, mode="sample"):  # Gaussian-kernel kNN similarity graph
    Z  = X if mode == "sample" else X.T
    m  = Z.shape[0]
    sq = _sq_dists(Z)
    idx = np.argsort(sq, axis=1)[:, 1:k + 1]
    S   = np.zeros((m, m))
    r   = np.repeat(np.arange(m), k)
    S[r, idx.ravel()] = np.exp(-sq[r, idx.ravel()] / sigma)
    return np.maximum(S, S.T)

def _laplacian(S):  # graph Laplacian L = D - S
    return np.diag(S.sum(1)) - S

def _omega(W, XtX, E_c):
    # Omega(W) = Tr(W' X' X W 1) + ||W||_{2,1} - ||W||_F
    return (float(np.trace(W.T @ XtX @ W @ E_c))
            + float(np.sum(np.linalg.norm(W, axis=1)))
            - float(np.linalg.norm(W)))

def _split(S):
    # Return (S_sym, degree_vector) so that L = diag(deg) - S_sym.
    # Multiplicative updates must never place the indefinite L in the
    # denominator: the affinity part S_sym is non-negative and belongs in the
    # numerator, the degree part in the denominator.
    S_sym = (S + S.T) / 2.0
    return S_sym, S_sym.sum(1)

# Baseline: use all features without selection
class Baseline:
    name = "Baseline"

    def fit(self, X):
        self.d = X.shape[1]
        return self

    def select(self, l):
        return np.arange(self.d)

# VarianceScore: select features with the largest variance (filter method)
class VarianceScore:
    name = "VarianceScore"

    def fit(self, X):
        self.scores = np.var(np.asarray(X, dtype=np.float64), axis=0)
        return self

    def select(self, l):
        return np.argsort(-self.scores)[:l]

# MCFS (Cai et al., 2010): spectral embedding + LASSO regression scores
class MCFS:
    """Multi-Cluster Feature Selection (Cai, Zhang & He, KDD 2010).

    For each leading eigenvector of the normalized graph Laplacian, a LARS
    path is computed. Feature scores are the maximum absolute coefficients
    across the paths, evaluated at the requested cardinality.
    """
    name = "MCFS"

    def __init__(self, c, k=5, sigma=1e4, n_eig=None):
        self.c, self.k, self.sigma = c, k, sigma
        self.n_eig = n_eig or c

    def fit(self, X):
        from sklearn.linear_model import lars_path
        X  = np.asarray(X, dtype=np.float64)
        S  = _knn_graph(X, self.k, self.sigma)
        d  = S.sum(1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, EPS))
        L_sym = np.eye(X.shape[0]) - d_inv_sqrt[:, None] * S * d_inv_sqrt
        _, vecs = np.linalg.eigh(L_sym)
        Y = vecs[:, 1:self.n_eig + 1]

        self._paths, self._nnz = [], []
        entry = np.full(X.shape[1], np.inf)
        for r in range(Y.shape[1]):
            _, _, coefs = lars_path(X, Y[:, r], method="lasso")
            nz = coefs != 0
            self._paths.append(coefs)
            self._nnz.append(nz.sum(0))
            first = np.where(nz.any(1), nz.argmax(1), np.inf)
            entry = np.minimum(entry, first)
        self._entry = entry
        self.n_ranked = int(np.isfinite(entry).sum())
        self.scores = np.max([np.abs(c_[:, -1]) for c_ in self._paths], axis=0)
        return self

    def select(self, l):
        l = int(min(l, len(self._entry)))
        a = np.zeros(len(self._entry))
        for coefs, nnz in zip(self._paths, self._nnz):
            hit = np.where(nnz >= l)[0]
            step = hit[0] if hit.size else coefs.shape[1] - 1
            a = np.maximum(a, np.abs(coefs[:, step]))
        order = np.lexsort((np.arange(len(a)), self._entry, -a))
        return order[:l]

# JELSR (Hou et al., 2014): joint embedding learning and sparse regression
class JELSR:
    name = "JELSR"

    def __init__(self, c, alpha=1.0, beta=1.0,
                 k=5, sigma=1e4, max_iter=50, tol=1e-4, seed=0):
        self.c     = c
        self.alpha, self.beta = alpha, beta
        self.k, self.sigma    = k, sigma
        self.max_iter, self.tol, self.seed = max_iter, tol, seed

    def fit(self, X):
        X  = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        L  = _laplacian(_knn_graph(X, self.k, self.sigma))
        _, vecs = np.linalg.eigh(L)
        F  = vecs[:, :self.c].copy()
        W  = np.zeros((d, self.c))
        XtX = X.T @ X
        A_f = np.eye(n) + self.alpha * L
        obj_prev = np.inf
        for _ in range(self.max_iter):

            D_w = np.diag(1.0 / (2.0 * np.maximum(
                np.linalg.norm(W, axis=1), EPS)))
            A_w = XtX + self.beta * D_w + 1e-8 * np.eye(d)
            try:    W = np.linalg.solve(A_w, X.T @ F)
            except: W = np.linalg.lstsq(A_w, X.T @ F, rcond=None)[0]

            try:    B = np.linalg.solve(A_f, X @ W)
            except: B = np.linalg.lstsq(A_f, X @ W, rcond=None)[0]
            U, _, Vt = np.linalg.svd(B, full_matrices=False)
            F = U @ Vt

            obj = (float(np.sum((X @ W - F) ** 2))
                   + self.alpha * float(np.trace(F.T @ L @ F))
                   + self.beta  * float(np.sum(np.linalg.norm(W, axis=1))))
            if abs(obj_prev - obj) / max(abs(obj_prev), EPS) < self.tol:
                break
            obj_prev = obj
        self.scores = np.linalg.norm(W, axis=1)
        return self

    def select(self, l):
        return np.argsort(-self.scores)[:l]

# NDFS (Li et al., 2012): nonnegative discriminative feature selection
class NDFS:
    name = "NDFS"

    def __init__(self, c, alpha=1.0, beta=1.0, lam=1e8,
                 k=5, sigma=1e4, max_iter=50, tol=1e-4, seed=0):
        self.c    = c
        self.alpha, self.beta, self.lam = alpha, beta, lam
        self.k, self.sigma = k, sigma
        self.max_iter, self.tol, self.seed = max_iter, tol, seed

    def fit(self, X):
        X  = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        L  = _laplacian(_knn_graph(X, self.k, self.sigma))
        rng = np.random.default_rng(self.seed)
        F  = rng.random((n, self.c)) + EPS
        W  = rng.random((d, self.c)) * 0.1
        XtX = X.T @ X
        I_c = np.eye(self.c)
        obj_prev = np.inf
        for _ in range(self.max_iter):
            D_w = np.diag(1.0 / (2.0 * np.maximum(
                np.linalg.norm(W, axis=1), EPS)))
            A   = self.alpha * XtX + self.beta * D_w + 1e-8 * np.eye(d)
            try:    W = np.linalg.solve(A, self.alpha * (X.T @ F))
            except: W = np.linalg.lstsq(A, self.alpha*(X.T@F), rcond=None)[0]
            num = self.alpha*(X@W) + 2.0*self.lam*F
            den = L@F + self.alpha*F + 2.0*self.lam*(F@(F.T@F))
            F   = np.clip(F * num / np.maximum(den, EPS), 0, CLIP_MAX)
            obj = (float(np.trace(F.T@L@F))
                   + self.alpha * float(np.sum((X@W-F)**2))
                   + self.beta  * float(np.sum(np.linalg.norm(W, axis=1)))
                   + self.lam   * float(np.sum((F.T@F-I_c)**2)))
            if abs(obj_prev - obj) / max(abs(obj_prev), EPS) < self.tol:
                break
            obj_prev = obj
        self.scores = np.linalg.norm(W, axis=1)
        return self

    def select(self, l):
        return np.argsort(-self.scores)[:l]

# UDLA (Wang et al., 2026): dual-graph clustering with a FIXED sample graph
class UDLA:
    name = "UDLA"

    def __init__(self, c, alpha=1.0, beta=1.0, gamma=1.0, lam=1.0,
                 k=5, sigma=1e4, max_iter=50, tol=1e-4, seed=0):
        self.c = c
        self.alpha, self.beta, self.gamma, self.lam = alpha, beta, gamma, lam
        self.k, self.sigma = k, sigma
        self.max_iter, self.tol, self.seed = max_iter, tol, seed

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        c    = self.c
        a, b, g, l = self.alpha, self.beta, self.gamma, self.lam

        S_X, dX = _split(_knn_graph(X, self.k, self.sigma, "sample"))
        S_V, dV = _split(_knn_graph(X, self.k, self.sigma, "feature"))  # fixed feature graph
        L_X = np.diag(dX) - S_X
        L_V = np.diag(dV) - S_V

        rng   = np.random.default_rng(self.seed)
        R     = rng.random((n, d)) + EPS
        B     = np.ones((d, c)) * 0.1 + rng.random((d, c)) * 0.01
        W     = np.ones((d, c)) * 0.1 + rng.random((d, c)) * 0.01
        F     = np.ones((c, n)) * 0.1 + rng.random((c, n)) * 0.01
        theta = np.ones(d) / d
        XtX   = X.T @ X
        E_c   = np.ones((c, c))
        obj_prev = np.inf

        for _ in range(self.max_iter):
            w_n  = np.linalg.norm(W, axis=1)
            Vd   = 1.0 / (2.0 * np.maximum(w_n, EPS))
            ist  = 1.0 / max(np.sqrt(np.sum(W * W)), EPS)
            t2   = theta ** 2

            XW_cur = X @ W
            nW = (X.T@R@B + a*(t2[:,None]*(X.T@F.T)) + l*0.5*ist*W
                  + g*(X.T@(S_X@XW_cur)))
            dW = (XtX@W  + a*(t2[:,None]*(W@(F@F.T)))
                  + g*(X.T@(dX[:,None]*XW_cur)) + l*(XtX@W@E_c + Vd[:,None]*W))
            W  = np.clip(W * nW / np.maximum(dW, EPS), 0, CLIP_MAX)

            w_col = np.linalg.norm(W, axis=0, keepdims=True)
            w_col[w_col < EPS] = 1.0
            W = W / w_col

            tW = t2[:, None] * W
            F  = np.clip(F*(tW.T@X.T)/np.maximum(tW.T@W@F, EPS), 0, CLIP_MAX)
            R  = np.clip(R*(X@W@B.T + b*(S_X@R))
                         / np.maximum(R@(B@B.T) + b*(dX[:,None]*R), EPS),
                         0, CLIP_MAX)
            B  = np.clip(B*(R.T@X@W + b*(S_V@B))
                         / np.maximum((R.T@R)@B + b*(dV[:,None]*B), EPS),
                         0, CLIP_MAX)

            diff  = X.T - W @ F
            q     = np.maximum(np.sum(diff * diff, axis=1), EPS)
            theta = (1.0 / q) / np.sum(1.0 / q)

            XW  = X @ W
            obj = (float(np.sum((XW-R@B)**2))
                   + a*float(np.sum((theta[:,None]*(X.T-W@F))**2))
                   + b*float(np.trace(R.T@L_X@R)+np.trace(B.T@L_V@B))
                   + g*float(np.trace(XW.T@L_X@XW))
                   + l*_omega(W, XtX, E_c))
            if abs(obj_prev - obj) / max(abs(obj_prev), EPS) < self.tol:
                break
            obj_prev = obj

        self.scores = np.linalg.norm(W, axis=1)
        return self

    def select(self, l):
        return np.argsort(-self.scores)[:l]

class AFAG:
    name = "AFAG"

    def __init__(self, c, alpha=1.0, beta=1.0, gamma=1.0,
                 lam=1.0, mu=1.0, k=5, sigma=1e4,
                 max_iter=50, tol=1e-4,
                 graph_damping=0.3, seed=0):
        self.c = c
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.lam, self.mu = lam, mu
        self.k, self.sigma = k, sigma
        self.max_iter, self.tol = max_iter, tol
        self.graph_damping = graph_damping
        self.seed = seed

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        c    = self.c
        a, b, g, l, mu = (self.alpha, self.beta, self.gamma,
                          self.lam,  self.mu)

        S_V, dV = _split(_knn_graph(X, self.k, self.sigma, "feature"))
        L_V = np.diag(dV) - S_V

        S_X = _knn_graph(X, self.k, self.sigma, "sample")  # warm-start sample graph
        S_sym, dX = _split(S_X)
        S_X = S_sym.copy()
        L_X = np.diag(dX) - S_sym

        rng   = np.random.default_rng(self.seed)
        R     = rng.random((n, d)) + EPS
        B     = np.ones((d, c)) * 0.1 + rng.random((d, c)) * 0.01
        W     = np.ones((d, c)) * 0.1 + rng.random((d, c)) * 0.01
        F     = np.ones((c, n)) * 0.1 + rng.random((c, n)) * 0.01
        theta = np.ones(d) / d
        XtX   = X.T @ X
        E_c   = np.ones((c, c))
        obj_prev = np.inf
        self.obj_history = []

        for it in range(self.max_iter):

            w_n  = np.linalg.norm(W, axis=1)
            Vd   = 1.0 / (2.0 * np.maximum(w_n, EPS))
            ist  = 1.0 / max(np.sqrt(np.sum(W * W)), EPS)
            t2   = theta ** 2
            XW_cur = X @ W
            nW = (X.T@R@B + a*(t2[:,None]*(X.T@F.T)) + l*0.5*ist*W
                  + g*(X.T@(S_sym@XW_cur)))
            dW = (XtX@W  + a*(t2[:,None]*(W@(F@F.T)))
                  + g*(X.T@(dX[:,None]*XW_cur)) + l*(XtX@W@E_c + Vd[:,None]*W))
            W  = np.clip(W * nW / np.maximum(dW, EPS), 0, CLIP_MAX)

            w_col = np.linalg.norm(W, axis=0, keepdims=True)
            w_col[w_col < EPS] = 1.0
            W = W / w_col

            tW = t2[:, None] * W
            F  = np.clip(F*(tW.T@X.T)/np.maximum(tW.T@W@F, EPS), 0, CLIP_MAX)

            R  = np.clip(R*(X@W@B.T + b*(S_sym@R))
                         / np.maximum(R@(B@B.T) + b*(dX[:,None]*R), EPS),
                         0, CLIP_MAX)

            B  = np.clip(B*(R.T@X@W + b*(S_V@B))
                         / np.maximum((R.T@R)@B + b*(dV[:,None]*B), EPS),
                         0, CLIP_MAX)

            diff  = X.T - W @ F
            q     = np.maximum(np.sum(diff * diff, axis=1), EPS)
            theta = (1.0 / q) / np.sum(1.0 / q)

            # adaptive update of the sample graph (the core of AFAG)
            Y       = X @ W
            S_new   = self._update_SX(R, Y, b, g, mu)
            S_X     = (S_new if it == 0 else
                       (1 - self.graph_damping) * S_X
                       + self.graph_damping * S_new)
            S_sym, dX = _split(S_X)
            L_X     = np.diag(dX) - S_sym

            XW  = X @ W
            obj = (float(np.sum((XW-R@B)**2))
                   + a*float(np.sum((theta[:,None]*(X.T-W@F))**2))
                   + b*float(np.trace(R.T@L_X@R)+np.trace(B.T@L_V@B))
                   + g*float(np.trace(XW.T@L_X@XW))
                   + l*_omega(W, XtX, E_c)
                   + mu*float(np.sum(S_X**2)))

            if not np.isfinite(obj):
                raise FloatingPointError(
                    f"AFAG objective became non-finite at iteration {it}. "
                    "This indicates a genuine numerical problem and must be "
                    "investigated, not silently suppressed.")
            self.obj_history.append(obj)

            if it > 0 and (obj_prev - obj) >= 0 and \
               abs(obj_prev - obj) / max(abs(obj_prev), EPS) < self.tol:
                break
            obj_prev = obj

        self.scores = np.linalg.norm(W, axis=1)

        return self

    def _update_SX(self, R, Y, beta, gamma, mu):
        # closed-form per-row update of S^X; mu controls the adaptive neighbour count
        n = R.shape[0]
        H = (beta / 2.0) * _sq_dists(R) + (gamma / 2.0) * _sq_dists(Y)
        H_ns = H.copy(); np.fill_diagonal(H_ns, np.inf)
        sH   = np.sort(H_ns, axis=1)[:, :n-1]
        cumH = np.cumsum(sH, axis=1)
        ks   = np.arange(1, n).reshape(1, -1)
        nus  = (2.0 * mu + cumH) / ks
        valid     = sH < nus
        k_per_row = np.where(valid.any(1), (valid*ks).max(1), 1).astype(int)
        idx       = np.arange(n)
        nu_row    = (2.0*mu + cumH[idx, k_per_row-1]) / k_per_row
        S = np.maximum((nu_row[:, None] - H) / (2.0 * mu), 0.0)
        np.fill_diagonal(S, 0.0)
        return S

    def select(self, l):
        return np.argsort(-self.scores)[:l]

# Clustering accuracy with Kuhn-Munkres optimal label assignment
def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    D = int(max(y_pred.max(), y_true.max()) + 1)
    cost = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        cost[y_pred[i], y_true[i]] += 1

    ri, ci = linear_sum_assignment(cost.max() - cost)
    return cost[ri, ci].sum() / len(y_pred)

# NMI with geometric-mean normalisation (as defined in UDLA)
def normalized_mutual_info_geom(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    I = mutual_info_score(y_true, y_pred)

    _, counts_pred = np.unique(y_pred, return_counts=True)
    _, counts_true = np.unique(y_true, return_counts=True)
    H_pred = entropy(counts_pred)
    H_true = entropy(counts_true)

    denom = np.sqrt(max(H_pred, EPS) * max(H_true, EPS))
    return float(I / denom)

def evaluate_clustering(X_sel, y_true, n_runs=20, seed=0):
    c = len(np.unique(y_true))
    accs, nmis = [], []
    for r in range(n_runs):
        km = KMeans(n_clusters=c, n_init=10, init="k-means++",
                    random_state=seed + r)
        yp = km.fit_predict(X_sel)
        accs.append(clustering_accuracy(y_true, yp))
        nmis.append(normalized_mutual_info_geom(y_true, yp))
    return {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "nmi_mean": float(np.mean(nmis)), "nmi_std": float(np.std(nmis)),
        "acc_runs": [float(v) for v in accs],
        "nmi_runs": [float(v) for v in nmis],
    }
