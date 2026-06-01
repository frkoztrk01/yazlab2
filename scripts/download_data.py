"""Download SKAB and BATADAL datasets."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_config, resolve_path


def download_skab(target_dir: Path | None = None) -> Path:
    target = target_dir or resolve_path(load_config(), "skab_raw")
    target.mkdir(parents=True, exist_ok=True)

    repo_dir = target / "_repo"
    if not (target / "valve1").exists() or not (target / "valve2").exists():
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/waico/skab.git", str(repo_dir)],
            check=True,
        )
        for valve in ("valve1", "valve2"):
            src = repo_dir / "data" / valve
            dst = target / valve
            if dst.exists():
                shutil.rmtree(dst)
            if src.exists():
                shutil.copytree(src, dst)
        shutil.rmtree(repo_dir, ignore_errors=True)

    return target


def download_batadal(target_dir: Path | None = None) -> Path:
    target = target_dir or resolve_path(load_config(), "batadal_raw")
    target.mkdir(parents=True, exist_ok=True)

    urls = {
        "BATADAL_dataset04.csv": "https://www.batadal.net/data/BATADAL_dataset04.csv",
    }

    for filename, url in urls.items():
        dest = target / filename
        if not dest.exists():
            print(f"Downloading {filename}...")
            try:
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                urlretrieve(url, dest, context=ctx)
            except Exception as exc:
                print(f"Warning: could not download {filename}: {exc}")

    return target


def main() -> None:
    print("Downloading SKAB...")
    skab_path = download_skab()
    print(f"SKAB ready at {skab_path}")

    print("Downloading BATADAL...")
    batadal_path = download_batadal()
    print(f"BATADAL ready at {batadal_path}")


if __name__ == "__main__":
    main()
