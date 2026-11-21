"""Download and extract the MovieLens-1M dataset into data/ml-1m/."""
from __future__ import annotations

import argparse
import io
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def _fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    contexts = []
    try:
        import certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    contexts.append(ssl.create_default_context())
    contexts.append(ssl._create_unverified_context())

    last_err = None
    for ctx in contexts:
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return resp.read()
        except urllib.error.URLError as err:
            last_err = err
            if not isinstance(getattr(err, "reason", None), ssl.SSLError):
                raise
    raise last_err  # type: ignore[misc]


def download_movielens(out_dir: str | Path = "data", force: bool = False) -> Path:
    out = Path(out_dir)
    target = out / "ml-1m"
    if (target / "ratings.dat").exists() and not force:
        print(f"MovieLens-1M already exists at {target} (use --force to re-download).")
        return target

    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens-1M from {URL} ...")
    raw = _fetch_bytes(URL)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(out)
    print(f"Extracted MovieLens-1M successfully -> {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MovieLens-1M dataset.")
    parser.add_argument("--out_dir", default="data", help="Output parent directory.")
    parser.add_argument("--force", action="store_true", help="Force re-download.")
    args = parser.parse_args()
    download_movielens(args.out_dir, force=args.force)
