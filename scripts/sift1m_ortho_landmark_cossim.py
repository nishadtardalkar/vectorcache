"""
SIFT1M -> 128-dim landmark cosine similarities.

Landmarks: 128 random vectors in R^128, orthonormalized via QR.
Each database/query vector x becomes y where
  y_i = cos(x, L_i) = <x_hat, L_i>
i.e. Y = X_hat @ L  with L columns unit and orthogonal.

Because L is a full orthonormal basis, this is an orthogonal change of basis:
angles between unit vectors are preserved (IP/cosine NN ranks identical).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def make_ortho_landmarks(dim: int, seed: int) -> np.ndarray:
    """Return L (dim x dim) with orthonormal columns."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q.astype(np.float32)


def landmark_cossim(X: np.ndarray, L: np.ndarray) -> np.ndarray:
    """X (n, d), L (d, m) unit columns -> (n, m) cosine similarities."""
    return l2_normalize(X.astype(np.float32, copy=False)) @ L


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hdf5",
        type=Path,
        default=Path("data/sift-128-euclidean.hdf5"),
    )
    p.add_argument("--n-landmarks", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-limit", type=int, default=0, help="0 = all train")
    p.add_argument("--query-limit", type=int, default=0, help="0 = all test")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/.cache/ortho_landmark_cossim"),
    )
    args = p.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        train = f["train"]
        test = f["test"]
        n_train = train.shape[0] if args.train_limit <= 0 else min(args.train_limit, train.shape[0])
        n_query = test.shape[0] if args.query_limit <= 0 else min(args.query_limit, test.shape[0])
        dim = int(train.shape[1])
        X = np.asarray(train[:n_train], dtype=np.float32)
        Q = np.asarray(test[:n_query], dtype=np.float32)

    if args.n_landmarks != dim:
        raise SystemExit(
            f"n_landmarks={args.n_landmarks} must equal dim={dim} for a full "
            "orthonormal landmark basis (square QR)."
        )

    L = make_ortho_landmarks(dim, args.seed)
    ortho_err = float(np.linalg.norm(L.T @ L - np.eye(dim, dtype=np.float32), ord="fro"))

    Y_train = landmark_cossim(X, L)
    Y_query = landmark_cossim(Q, L)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "landmarks.npy", L)
    np.save(args.out_dir / "train_cossim.npy", Y_train)
    np.save(args.out_dir / "query_cossim.npy", Y_query)

    meta = {
        "hdf5": str(args.hdf5),
        "n_train": n_train,
        "n_query": n_query,
        "dim": dim,
        "n_landmarks": args.n_landmarks,
        "seed": args.seed,
        "orthogonality_error": ortho_err,
        "train_cossim_shape": list(Y_train.shape),
        "query_cossim_shape": list(Y_query.shape),
        "train_row_norm_mean": float(np.linalg.norm(Y_train, axis=1).mean()),
        "query_row_norm_mean": float(np.linalg.norm(Y_query, axis=1).mean()),
        "train_cossim_mean": float(Y_train.mean()),
        "train_cossim_std": float(Y_train.std()),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out_dir}/landmarks.npy")
    print(f"wrote {args.out_dir}/train_cossim.npy")
    print(f"wrote {args.out_dir}/query_cossim.npy")


if __name__ == "__main__":
    main()
