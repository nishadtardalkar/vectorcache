#!/usr/bin/env python3
"""Download GloVe HDF5 (if missing) and extract a float32 NPY sample for ingestion."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np

GLOVE_URL = "http://ann-benchmarks.com/glove-200-angular.hdf5"
GLOVE_FILENAME = "glove-200-angular.hdf5"
MIN_BYTES = 100_000_000


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Downloading {url} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "vectorcache/0.1"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    print(f"Saved {dest}")


def ensure_glove(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / GLOVE_FILENAME
    if dest.is_file() and dest.stat().st_size >= MIN_BYTES:
        print(f"Using existing {dest}")
        return dest
    download(GLOVE_URL, dest)
    if dest.stat().st_size < MIN_BYTES:
        raise SystemExit(f"downloaded file too small: {dest}")
    return dest


def extract_sample(hdf5_path: Path, out_path: Path, limit: int, split: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "r") as f:
        if split not in f:
            raise SystemExit(f"missing '{split}' dataset in {hdf5_path}")
        ds = f[split]
        rows = min(limit, ds.shape[0])
        sample = ds[:rows].astype(np.float32, copy=False)
    np.save(out_path, sample)
    print(f"Wrote {rows} x {sample.shape[1]} vectors to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--out",
        type=Path,
        help="output .npy path (default: data/.cache/glove-sample-<limit>.npy)",
    )
    args = parser.parse_args()

    glove_path = ensure_glove(args.data_dir)
    out_path = args.out or args.data_dir / ".cache" / f"glove-sample-{args.limit}.npy"
    extract_sample(glove_path, out_path, args.limit, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
