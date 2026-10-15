"""
Start with N random N-dimensional vectors and make them orthonormal.

Best algorithm: QR decomposition (A = QR). The columns of Q are an
orthonormal basis spanning the same space as the input columns. This is
the numerically stable way to do Gram-Schmidt (Householder QR under the hood
in LAPACK/numpy).

Alternatives:
  - Modified Gram-Schmidt: fine if you want an explicit loop; less stable than QR
  - Classic Gram-Schmidt: avoid for N >> 10; loses orthogonality to float error
  - SVD: also works, overkill if you only need any orthonormalization of the span
"""

from __future__ import annotations

import argparse

import numpy as np


def orthogonalize_qr(A: np.ndarray) -> np.ndarray:
    """Orthonormalize columns of A via thin QR. Returns Q (N x N) with Q.T @ Q = I."""
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q


def orthogonalize_mgs(A: np.ndarray) -> np.ndarray:
    """Modified Gram-Schmidt (educational). Prefer QR in practice."""
    n = A.shape[1]
    Q = A.astype(np.float64, copy=True)
    for i in range(n):
        Q[:, i] /= np.linalg.norm(Q[:, i])
        for j in range(i + 1, n):
            Q[:, j] -= np.dot(Q[:, i], Q[:, j]) * Q[:, i]
    return Q


def orthogonality_error(Q: np.ndarray) -> float:
    """||Q.T @ Q - I||_F — should be ~1e-15 for QR."""
    n = Q.shape[1]
    return float(np.linalg.norm(Q.T @ Q - np.eye(n), ord="fro"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", type=int, default=8, help="number of vectors / dimensions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--method",
        choices=("qr", "mgs"),
        default="qr",
        help="qr (recommended) or modified Gram-Schmidt",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    # Columns are the N vectors in R^N
    A = rng.standard_normal((args.n, args.n))

    Q = orthogonalize_qr(A) if args.method == "qr" else orthogonalize_mgs(A)

    print(f"N={args.n} method={args.method}")
    print(f"orthogonality error (Frobenius): {orthogonality_error(Q):.3e}")
    print(f"column norms: {np.linalg.norm(Q, axis=0)}")
    # Spot-check a few pairwise dots
    if args.n >= 2:
        print(f"Q[:,0]·Q[:,1] = {np.dot(Q[:, 0], Q[:, 1]):.3e}")


if __name__ == "__main__":
    main()
