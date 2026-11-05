"""Adaptive-graph feature-selection methods and shared graph utilities.

The module provides AFAG, UDLA, CNAFS, SOGFS, and MGAGR with a common
``fit(X)`` / ``select(l)`` interface. CNAFS and MGAGR are based on their
authors' reference implementations; SOGFS follows its published algorithms.
The CNAFS and MGAGR sources are credited in their class docstrings.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

from algorithms import AFAG, UDLA          # re-exported, see __all__

__all__ = ["AFAG", "UDLA", "CNAFS", "SOGFS", "MGAGR"]

EPS = 1e-10


# ------------------------------------------------------------------ helpers
def _sq_dists(Z):
    """Pairwise squared Euclidean distances between the rows of Z."""
    return squareform(pdist(np.asarray(Z, dtype=np.float64), "sqeuclidean"))


def mu_from_k(D, k, exclude_diag=True):
    """Eq. (37) of the SOGFS journal paper: pick the quadratic weight so that
    most rows end up with about k non-zero entries.  Returns mu such that the
    solver below yields k_i ~ k."""
    D = np.asarray(D, dtype=np.float64).copy()
    if exclude_diag:
        np.fill_diagonal(D, np.inf)
    Ds = np.sort(D, axis=1)
    k = int(min(k, Ds.shape[1] - 1))
    # alpha_i = (k/2) d_{i,k+1} - (1/2) sum_{j<=k} d_ij
    a = 0.5 * (k * Ds[:, k] - Ds[:, :k].sum(1))
    a = a[np.isfinite(a)]
    return float(max(np.mean(a), EPS)) if a.size else 1.0


def _can_rows(D, mu, exclude_diag=True):
    """Row-wise solution of  min_{s>=0, sum s = 1} sum_j (D_ij s_ij + mu s_ij^2).

    Closed form  s_ij = max((nu_i - D_ij) / (2 mu), 0)  with the support size
    k_i = min{k : g_i(k) >= 2 mu},  g_i(k) = k*D_{i,(k+1)} - sum_{j<=k} D_{i,(j)}.
    g_i is non-decreasing in k, so k_i is unique.
    """
    D = np.asarray(D, dtype=np.float64).copy()
    n = D.shape[0]
    if exclude_diag:
        np.fill_diagonal(D, np.inf)

    order = np.argsort(D, axis=1)
    Ds = np.take_along_axis(D, order, axis=1)
    m = Ds.shape[1] - (1 if exclude_diag else 0)        # usable candidates

    ks = np.arange(1, m)                                 # k = 1 .. m-1
    csum = np.cumsum(Ds[:, :m], axis=1)
    g = ks * Ds[:, 1:m] - csum[:, :m - 1]                # g_i(k), non-decreasing
    hit = g >= 2.0 * mu
    k_i = np.where(hit.any(1), hit.argmax(1) + 1, m)     # fall back to all

    S = np.zeros((n, n))
    rows = np.arange(n)
    for kk in np.unique(k_i):
        sel = rows[k_i == kk]
        nu = (2.0 * mu + csum[sel, kk - 1]) / kk         # Eq. for nu_i
        vals = np.maximum((nu[:, None] - Ds[sel, :kk]) / (2.0 * mu), 0.0)
        ssum = vals.sum(1, keepdims=True)
        vals = vals / np.maximum(ssum, EPS)              # guard against ties
        # NB: S[sel] would be a copy under advanced indexing, so build the
        # rows first and assign them back explicitly.
        block = np.zeros((sel.size, n))
        np.put_along_axis(block, order[sel, :kk], vals, axis=1)
        S[sel] = block
    if exclude_diag:
        np.fill_diagonal(S, 0.0)
    return S


def _sym_lap(S):
    """Symmetrised affinity, degree vector and Laplacian L = diag(deg) - S_sym."""
    S_sym = (S + S.T) / 2.0
    deg = S_sym.sum(1)
    return S_sym, deg, np.diag(deg) - S_sym


def _smallest_eigvecs(A, m):
    """The m eigenvectors of the symmetric matrix A with smallest eigenvalues."""
    A = (A + A.T) / 2.0
    m = int(min(m, A.shape[0]))
    _, V = eigh(A, subset_by_index=[0, m - 1])
    return V


def _gpi(A, B, n_iter=5, seed=0):
    """Generalized power iteration for  min_{W'W = I} Tr(W'AW - 2W'B).

    Nie, Zhang & Li, Science China Information Sciences 60, 112101 (2017).
    Ported from gpi.m in the CNAFS reference implementation.
    """
    A = (A + A.T) / 2.0
    m, k = B.shape
    if m < k:
        raise ValueError(f"_gpi: need m >= k, got {m} < {k}")

    rng = np.random.default_rng(seed)
    w = rng.random(m)
    for _ in range(10):                                  # power iteration
        w = A @ w
        nw = np.linalg.norm(w)
        if nw < EPS:
            break
        w = w / nw
    alpha = abs(float(w @ A @ w))

    W = np.linalg.qr(rng.random((m, k)))[0]
    A_til = alpha * np.eye(m) - A
    for _ in range(n_iter):
        M = A_til @ W + 2.0 * B
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        W = U @ Vt
    return W


def _check_dim(d, limit, name):
    if d > limit:
        raise MemoryError(
            f"{name} needs dense d x d algebra (d={d}); at d>{limit} this "
            f"exceeds a reasonable memory/time budget. Report '--' for this "
            f"dataset and state the reason, or lower the limit deliberately "
            f"via max_dim.")


# ------------------------------------------------------------------- CNAFS
class CNAFS:
    """Convex NMF with adaptive graph (Yuan et al., T-Cybernetics 2022).

    Port of CNAFS.m.  The model is  X ~= X G V  (convex NMF, so the basis is a
    convex combination of samples) together with a pseudo-label regression
    H(X'W - F) and an adaptive graph S built from BOTH the pseudo-label matrix
    F and the coefficient matrix V.  Feature scores are the row norms of W.

    Parameters follow the reference code: alpha, beta, lam, gamma, epsilon.
    nmf_k is the number of convex-NMF basis vectors (reference argument NMF_K);
    it defaults to the cluster number.
    """
    name = "CNAFS"

    def __init__(self, c, alpha=1.0, beta=1.0, lam=1.0, gamma=1.0,
                 epsilon=1e-4, max_iter=30, nmf_k=None, tol=0.1, seed=0,
                 max_dim=None):
        self.c, self.alpha, self.beta = c, alpha, beta
        self.lam, self.gamma, self.epsilon = lam, gamma, epsilon
        self.max_iter, self.nmf_k, self.tol = max_iter, nmf_k, tol
        self.seed, self.max_dim = seed, max_dim

    def fit(self, X):
        Xn = np.asarray(X, dtype=np.float64)             # (n, d) as in project
        n, d = Xn.shape
        if self.max_dim:
            _check_dim(d, self.max_dim, "CNAFS")
        Xm = Xn.T                                        # (d, n) as in CNAFS.m
        c = self.c
        K = int(self.nmf_k or c)
        a, b, lam, g, eps = (self.alpha, self.beta, self.lam,
                             self.gamma, self.epsilon)
        rng = np.random.default_rng(self.seed)

        H = np.eye(n) - np.ones((n, n)) / n              # centering
        F = np.linalg.qr(rng.random((n, c)))[0]
        W = rng.random((d, c))
        G = np.abs(rng.random((n, K)))
        V = np.abs(rng.random((K, n)))
        Q = np.ones((K, K)) - np.eye(K)                  # redundancy penalty

        XHXt = Xm @ H @ Xm.T                             # (d, d), fixed
        XtX = Xn @ Xn.T                                  # = Xm.T @ Xm, (n, n)

        self.obj_history = []
        obj_prev = np.inf
        for _ in range(self.max_iter):
            D_w = 0.5 / np.sqrt(np.sum(W * W, axis=1) + eps)   # diag of D_weight

            # adaptive graph from the pseudo-labels F and the coefficients V
            M = a * _sq_dists(F) + 0.5 * g * _sq_dists(V.T)
            S = np.exp(-M / (2.0 * a * b))
            S = S / np.maximum(S.sum(1, keepdims=True), EPS)
            S_sym, deg, L = _sym_lap(S)

            # W: closed form ridge-like solve (reference uses inv(); solve() is
            # the same system but numerically better behaved)
            W = np.linalg.solve(XHXt + lam * np.diag(D_w), Xm @ H @ F)
            W = np.real(W)

            # F: Stiefel-constrained quadratic, solved by GPI
            F = _gpi(H + 2.0 * a * L, H @ Xm.T @ W, seed=self.seed)

            # convex-NMF factors
            G = G * (XtX @ V.T) / np.maximum(XtX @ G @ (V @ V.T), EPS)
            V = V * ((G.T @ XtX + g * (V @ S_sym))
                     / np.maximum(G.T @ XtX @ G @ V
                                  + g * (V * deg[None, :]) + eps * (Q @ V), EPS))

            ent = float(np.sum(S * np.log(S + 1e-10)))   # negative entropy term
            obj = (float(np.sum((Xm - Xm @ G @ V) ** 2))
                   + float(np.sum((H @ (Xm.T @ W - F)) ** 2))
                   + lam * float(np.sum(D_w * np.sum(W * W, axis=1)))
                   + 2.0 * a * (float(np.trace(F.T @ L @ F)) + b * ent)
                   + g * float(np.trace(V @ L @ V.T))
                   + eps * float(np.trace(V.T @ Q @ V)))
            self.obj_history.append(obj)
            if abs(obj_prev - obj) <= self.tol:
                break
            obj_prev = obj

        self.W = W
        self.scores = np.sum(W * W, axis=1)
        return self

    def select(self, l):
        return np.argsort(-self.scores)[:l]


# ------------------------------------------------------------------- SOGFS
class SOGFS:
    """Structured Optimal Graph Feature Selection (Nie, Zhu & Li).

    Implemented from Algorithm 1 and Algorithm 2 (AAAI 2016 / TKDE 2021); there
    is no official release.  Jointly learns an orthogonal projection W with
    l_2,1 row sparsity and a similarity matrix S constrained to have exactly c
    connected components, the latter enforced through the rank-relaxation term
    2*lam*Tr(F'L_S F) with lam adapted each iteration.

    Feature scores are ||w_i||_2.

    Cost warning: step 1 needs the m smallest eigenvectors of a dense d x d
    matrix, i.e. O(d^3) per iteration, which is the binding constraint on
    high-dimensional data.
    """
    name = "SOGFS"

    def __init__(self, c, gamma=1.0, k=5, m=None, lam=1.0, max_iter=20,
                 inner_iter=1, eps=1e-8, tol=1e-5, seed=0, max_dim=6000):
        self.c, self.gamma, self.k, self.m = c, gamma, k, m
        self.lam, self.max_iter, self.inner_iter = lam, max_iter, inner_iter
        self.eps, self.tol, self.seed, self.max_dim = eps, tol, seed, max_dim

    def fit(self, X):
        Xn = np.asarray(X, dtype=np.float64)
        n, d = Xn.shape
        _check_dim(d, self.max_dim, "SOGFS")
        c = self.c
        m = int(self.m or c)
        lam = float(self.lam)

        # Algorithm 2, initialisation: S from problem (2) on the raw features,
        # with the quadratic weight set by Eq. (37) from the target k.
        D0 = _sq_dists(Xn)
        mu = mu_from_k(D0, self.k)
        S = _can_rows(D0, mu)

        Q = np.ones(d)                                   # diag of Q, init I
        W = None
        self.obj_history = []
        obj_prev = np.inf

        for _ in range(self.max_iter):
            S_sym, deg, L = _sym_lap(S)

            # step 1 -- update W (Algorithm 1)
            XLX = Xn.T @ L @ Xn
            for _ in range(self.inner_iter):
                W = _smallest_eigvecs(XLX + self.gamma * np.diag(Q), m)
                rn = np.sqrt(np.sum(W * W, axis=1) + self.eps)
                Q = 1.0 / (2.0 * rn)                     # Eq. (12)

            # step 3 -- update F: c smallest eigenvectors of L_S
            F = _smallest_eigvecs(L, c)

            # step 4 -- update S from the fused distance d_ij = m_ij + lam n_ij
            XW = Xn @ W
            Dfuse = _sq_dists(XW) + lam * _sq_dists(F)
            S = _can_rows(Dfuse, mu)

            # adapt lam so that S ends up with exactly c connected components
            ncomp = connected_components((S + S.T) / 2.0 > EPS,
                                         directed=False, return_labels=False)
            if ncomp < c:
                lam *= 2.0
            elif ncomp > c:
                lam /= 2.0

            obj = (float(np.trace(W.T @ XLX @ W))
                   + self.gamma * float(np.sum(np.sqrt(
                       np.sum(W * W, axis=1) + self.eps)))
                   + 2.0 * lam * float(np.trace(F.T @ L @ F)))
            self.obj_history.append(obj)
            if (np.isfinite(obj_prev)
                    and abs(obj_prev - obj) <= self.tol * max(abs(obj_prev), 1.0)):
                break
            obj_prev = obj

        self.W, self.S, self.lam_final = W, S, lam
        self.scores = np.linalg.norm(W, axis=1)
        return self

    def select(self, l):
        return np.argsort(-self.scores)[:l]


# ------------------------------------------------------------------- MGAGR
class MGAGR:
    """Robust UFS via Multi-Group Adaptive Graph Representation (You et al.).

    Port of MGAGR.m.  Features are partitioned into groups; one similarity
    graph is built per group, and a global graph S is learned as an adaptively
    weighted combination of them (weights W, one column per sample) jointly
    with a feature self-representation Z (X ~= X Z) carrying l_2,1 row
    sparsity.  Feature scores are the row norms of Z.

    The reference method takes feature-group labels as input. This
    implementation supports contiguous and seeded-random group assignments.
    The number of inner updates for Z is controlled by ``inner_iter``.

    Cost warning: Z is d x d and each update solves a dense d x d system, i.e.
    O(d^3) time and O(d^2) memory.
    """
    name = "MGAGR"

    def __init__(self, c=None, alpha=1.0, gamma=1.0, sigma=1e4, group_num=3,
                 group_mode="contiguous", max_iter=10, inner_iter=3,
                 eps=1e-15, seed=0, max_dim=6000):
        self.c, self.alpha, self.gamma, self.sigma = c, alpha, gamma, sigma
        self.group_num, self.group_mode = group_num, group_mode
        self.max_iter, self.inner_iter = max_iter, inner_iter
        self.eps, self.seed, self.max_dim = eps, seed, max_dim

    def _groups(self, d):
        v = int(self.group_num)
        if self.group_mode == "contiguous":
            return np.minimum((np.arange(d) * v) // d, v - 1)
        if self.group_mode == "random":
            g = np.arange(d) % v
            np.random.default_rng(self.seed).shuffle(g)
            return g
        raise ValueError(f"unknown group_mode {self.group_mode!r}")

    def fit(self, X):
        Xn = np.asarray(X, dtype=np.float64)
        n, d = Xn.shape
        _check_dim(d, self.max_dim, "MGAGR")
        v = int(self.group_num)
        a, g, eps = self.alpha, self.gamma, self.eps
        rng = np.random.default_rng(self.seed)
        glab = self._groups(d)

        def _gauss_rows(Z):
            S = np.exp(-_sq_dists(Z) / self.sigma)
            S = S / np.maximum(S.sum(1, keepdims=True), EPS)
            return (S + S.T) / 2.0

        S_d = [_gauss_rows(Xn[:, glab == q]) for q in range(v)]   # per group
        S = _gauss_rows(Xn)                                       # global
        Wg = np.ones((v, n)) / v
        Z = rng.random((d, d))

        self.obj_history = []
        for _ in range(self.max_iter):
            # ---- update the group weights, one column per sample
            for i in range(n):
                A_i = np.column_stack([S[:, i] - S_d[q][:, i] for q in range(v)])
                Bq = A_i.T @ A_i
                one = np.ones(v)
                try:
                    t = np.linalg.solve(Bq, one)
                except np.linalg.LinAlgError:
                    t = np.linalg.lstsq(Bq, one, rcond=None)[0]
                Wg[:, i] = t / (one @ t + eps)

            # ---- update the global graph S
            XZ = Xn @ Z
            Dz = _sq_dists(XZ)
            Bmat = np.stack([S_d[q] for q in range(v)], axis=2)   # (n, n, v)
            for i in range(n):
                part_m = Bmat[:, i, :] @ Wg[:, i] - 0.25 * a * Dz[i]
                # ones(n,n) @ part_m / n is the mean of part_m, repeated
                tmp = part_m - part_m.mean()
                psi = np.maximum(-2.0 * tmp, 0.0)
                tmp = part_m - part_m.mean() + 0.5 * psi.mean()
                row = np.maximum(tmp, 0.0)
                S[i] = row / max(row.sum(), EPS)
            S = (S + S.T) / 2.0

            # ---- update the self-representation Z
            _, _, L = _sym_lap(S)
            XtMX = Xn.T @ (np.eye(n) + a * L) @ Xn
            XtX = Xn.T @ Xn
            for _ in range(self.inner_iter):
                rn = 2.0 * np.sqrt(np.sum(Z * Z, axis=1)) + eps
                Z = np.linalg.solve(XtMX + g * np.diag(1.0 / rn), XtX)

            f1 = float(np.sum([
                np.sum((S[:, i] - Bmat[:, i, :] @ Wg[:, i]) ** 2)
                for i in range(n)]))
            f2 = float(np.sum((Xn - Xn @ Z) ** 2))
            f3 = g * float(np.sum(np.sqrt(np.sum(Z * Z, axis=1))))
            f4 = a * float(np.trace(Z.T @ Xn.T @ L @ Xn @ Z))
            self.obj_history.append(f1 + f2 + f3 + f4)

        self.Z, self.S, self.group_weights = Z, S, Wg
        self.scores = np.sum(Z * Z, axis=1)
        return self

    def select(self, l):
        return np.argsort(-self.scores)[:l]
