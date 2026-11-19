"""Scratch: per-query ONB via transporting original landmarks; score correction.

Score-time inputs: stored out coords + 1 branch bit each + global (L, DELTA).
a_hat / full bases are rebuilt on the fly (not stored).

Note: out + globals alone is not enough — the recovery quadratic has 2 roots;
one bit selects the encode-consistent root.
"""

import numpy as np

DELTA = 0.01


def random_onb(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Return (dim, dim) orthonormal basis as columns."""
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    d = np.sign(np.diag(r))
    d[d == 0] = 1.0
    q = q * d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def rotate_toward(v: np.ndarray, u_hat: np.ndarray, delta: float) -> np.ndarray:
    """Rotate v in the (u_hat, v) plane so <v', u_hat> = <v, u_hat> + delta."""
    a = float(np.dot(v, u_hat))
    nrm = float(np.linalg.norm(v))
    target = float(np.clip(a + delta, -nrm, nrm))
    perp = v - a * u_hat
    p = float(np.linalg.norm(perp))
    if p < 1e-12:
        return v.copy()
    n_hat = perp / p
    new_p = float(np.sqrt(max(nrm * nrm - target * target, 0.0)))
    return target * u_hat + new_p * n_hat


def plane_rotation_u_to_v(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unique rotation in span{u,v} sending unit(u) -> unit(v); I on the complement."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    dim = u.shape[0]
    c = float(np.dot(u, v))
    if c > 1.0 - 1e-12:
        return np.eye(dim)
    if c < -1.0 + 1e-12:
        w = np.zeros(dim)
        w[int(np.argmin(np.abs(u)))] = 1.0
        w = w - np.dot(w, u) * u
        w /= np.linalg.norm(w)
        return np.eye(dim) - 2.0 * np.outer(u, u) - 2.0 * np.outer(w, w)
    s = float(np.sqrt(max(1.0 - c * c, 0.0)))
    n = (v - c * u) / s
    return (
        np.eye(dim)
        + (c - 1.0) * (np.outer(u, u) + np.outer(n, n))
        + s * (np.outer(n, u) - np.outer(u, n))
    )


def transport_onb(L: np.ndarray, a_hat: np.ndarray) -> np.ndarray:
    """New ONB = R L, where R is the unique plane rot with R L[:,0] = a_hat."""
    return plane_rotation_u_to_v(L[:, 0], a_hat) @ L


def a_hat_candidates(coords: np.ndarray, L: np.ndarray, delta: float) -> list[np.ndarray]:
    """Two a_hat roots consistent with coords under (L, delta). Plane = span{u0, coords@L.T}."""
    u0 = L[:, 0]
    q0 = coords @ L.T
    nrm = float(np.linalg.norm(q0))
    alpha = float(np.dot(q0, u0))
    perp = q0 - alpha * u0
    beta = float(np.linalg.norm(perp))
    if beta < 1e-12:
        return [u0.copy(), u0.copy()]
    n = perp / beta
    # A c + B s = delta, c^2 + s^2 = 1, a = c u0 + s n
    A, B = nrm - alpha, beta
    aa = A * A + B * B
    bb = -2.0 * A * delta
    cc = delta * delta - B * B
    disc = max(bb * bb - 4.0 * aa * cc, 0.0)
    root = np.sqrt(disc)
    out = []
    for sgn in (+1.0, -1.0):
        c = (-bb + sgn * root) / (2.0 * aa)
        s = (delta - A * c) / B
        out.append(c * u0 + s * n)
    return out


def reconstruct_ambient(coords: np.ndarray, L: np.ndarray, delta: float, branch: int) -> np.ndarray:
    """Undo projection: rebuild Li from (out, branch, L, delta), return out @ Li.T."""
    a_hat = a_hat_candidates(coords, L, delta)[branch]
    Li = transport_onb(L, a_hat)
    return coords @ Li.T


def main() -> None:
    rng = np.random.default_rng(0)
    dim = 4
    nq = 10000

    Q = rng.normal(size=(nq, dim))
    L = random_onb(dim, rng)
    u0 = L[:, 0]

    out = np.empty_like(Q)
    branch = np.empty(nq, dtype=np.int8)
    for i in range(nq):
        r = rotate_toward(Q[i], u0, DELTA)
        a_true = r / np.linalg.norm(r)
        Li = transport_onb(L, a_true)
        out[i] = Q[i] @ Li
        cands = a_hat_candidates(out[i], L, DELTA)
        branch[i] = int(np.argmin([np.linalg.norm(c - a_true) for c in cands]))

    print("q:")
    print(np.round(Q[:5], 4))
    print("out:")
    print(np.round(out[:5], 4))
    print("bit:", branch[:5].tolist())

    # --- scoring ---
    # Actually expected inside the n^2 loop, for each pair (i, j):
    #   Li = transport_onb(L, a_hat_candidates(out[i], L, DELTA)[branch[i]])
    #   Lj = transport_onb(L, a_hat_candidates(out[j], L, DELTA)[branch[j]])
    #   ip = out[i] @ (Li.T @ Lj) @ out[j]
    #   D[i,j] = sqrt(||out[i]||^2 + ||out[j]||^2 - 2*ip)
    # That equals ||qi - qj|| with qi = out[i] @ Li.T, so we reconstruct ambient
    # once per vec (still the same n^2-loop math) then vectorize distances.
    Q_rec = np.stack(
        [reconstruct_ambient(out[i], L, DELTA, int(branch[i])) for i in range(nq)]
    )
    D0 = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1)
    D = np.linalg.norm(Q_rec[:, None, :] - Q_rec[None, :, :], axis=-1)

    top_k = min(64, nq - 1)
    order0 = np.argsort(D0, axis=1)
    order1 = np.argsort(D, axis=1)
    # drop self (distance 0 sorts first)
    order0 = order0[:, 1:]
    order1 = order1[:, 1:]

    rank_frac = float(np.mean(order0 == order1))
    topk_overlap = float(
        np.mean(
            [
                len(set(order0[i, :top_k].tolist()) & set(order1[i, :top_k].tolist())) / top_k
                for i in range(nq)
            ]
        )
    )
    print(f"mean rank order match: {100.0 * rank_frac:.2f}%")
    print(f"top-{top_k} overlap: {100.0 * topk_overlap:.2f}%")


if __name__ == "__main__":
    main()
