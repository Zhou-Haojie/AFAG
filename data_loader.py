"""Load the six benchmark datasets from MATLAB .mat files."""

import os
import numpy as np
import scipy.io as sio

def _find(dataset_dir, *candidates):
    for name in candidates:
        for suffix in ("", ".mat"):
            path = os.path.join(dataset_dir, name + suffix)
            if os.path.isfile(path):
                return path
    return None

def _load(path, dataset_name):
    print(f"  Loading {dataset_name} from {os.path.basename(path)} ...")
    mat = sio.loadmat(path)

    X = None
    for key in ("X", "fea", "data", "Data", "features"):
        if key in mat:
            arr = np.array(mat[key], dtype=np.float64)
            if arr.ndim == 2 and min(arr.shape) > 1:
                X = arr
                break

    y = None
    for key in ("Y", "gnd", "label", "labels", "Label", "y", "gt"):
        if key in mat:
            y = np.array(mat[key], dtype=int).flatten()
            break

    if X is None or y is None:
        keys = [k for k in mat if not k.startswith("__")]
        for k in keys:
            v = mat[k]
            if not hasattr(v, "shape"):
                continue
            if X is None and v.ndim == 2 and min(v.shape) > 5:
                X = np.array(v, dtype=np.float64)
            elif y is None and v.ndim <= 2 and v.size < 10000:
                y = np.array(v, dtype=int).flatten()

    if X is None or y is None:
        keys = [k for k in mat if not k.startswith("__")]
        raise ValueError(
            f"Cannot parse {dataset_name}. Available keys: {keys}")

    if X.shape[0] != y.size and X.shape[1] == y.size:
        X = X.T

    unique = sorted(set(y.tolist()))
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.array([label_map[int(v)] for v in y])
    c = len(unique)

    print(f"    → n={X.shape[0]}, d={X.shape[1]}, c={c}")
    return X, y, c

def load_warpar10p(dataset_dir):
    path = _find(dataset_dir, "warpAR10P", "warpar10p", "warpAR10p")
    if path is None:
        raise FileNotFoundError(
            f"warpAR10P.mat not found in {dataset_dir}")
    return _load(path, "warpAR10P")

def load_orl(dataset_dir):
    path = _find(dataset_dir, "ORL", "orl", "ORL_32x32")
    if path is None:
        raise FileNotFoundError(
            f"ORL.mat not found in {dataset_dir}")
    return _load(path, "ORL")

def load_yale(dataset_dir):
    path = _find(dataset_dir, "Yale", "yale", "Yale_32x32")
    if path is None:
        raise FileNotFoundError(
            f"Yale.mat not found in {dataset_dir}")
    return _load(path, "Yale")

def load_tox171(dataset_dir):
    path = _find(dataset_dir, "TOX-171", "TOX_171", "tox171", "TOX171")
    if path is None:
        raise FileNotFoundError(
            f"TOX-171.mat not found in {dataset_dir}")
    return _load(path, "TOX-171")

def load_cllsub111(dataset_dir):
    path = _find(dataset_dir,
                 "CLL-SUB-111", "CLL_SUB_111",
                 "CLL_SUB-111", "CLL-SUB_111",
                 "cll_sub_111", "cllsub111", "CLLSUB111")
    if path is None:
        raise FileNotFoundError(
            f"CLL-SUB-111.mat not found in {dataset_dir}")
    return _load(path, "CLL-SUB-111")

def load_glioma(dataset_dir):
    path = _find(dataset_dir, "GLIOMA", "glioma", "Glioma")
    if path is None:
        raise FileNotFoundError(
            f"GLIOMA.mat not found in {dataset_dir}")
    return _load(path, "GLIOMA")

LOADERS = {
    "warpAR10P":    load_warpar10p,
    "ORL":          load_orl,
    "Yale":         load_yale,
    "TOX-171":      load_tox171,
    "CLL-SUB-111":  load_cllsub111,
    "GLIOMA":       load_glioma,
}
