#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import openmedres as medres_module
import requests
from bs4 import BeautifulSoup  # might need: pip install beautifulsoup4

BASE_URL = "https://phoenix.astro.physik.uni-goettingen.de/data/v1.0/MedResFITS/R10000FITS/"

def list_all_zip_urls(base_url=BASE_URL):
    """Return every ZIP URL"""
    r = requests.get(base_url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if href.lower().endswith(".zip"):
            urls.append(base_url + href)
    return sorted(set(urls))

def convert_with_medres(medres_module, fits_path: Path, out_dir: Path):
    """Run medres.main() on one FITS -> TXT."""
    out_name = fits_path.name
    if "HiRes" in out_name:
        out_name = out_name.replace("HiRes", "MedRes") # For some reason, PHOENIX names R100000 MedRes files with HiRes
    elif "R10000FITS" in out_name:
        out_name = out_name.replace("R10000FITS", "MedRes")
    out_name = re.sub(r"\.fits$", ".txt", out_name, flags=re.IGNORECASE)
    out_path = out_dir / out_name

    old_argv = sys.argv[:]
    try:
        sys.argv = ["openmedres.py", str(fits_path), "--out", str(out_path)]
        medres_module.main()
    finally:
        sys.argv = old_argv
    return out_path

def download_and_process_zip(zip_url: str, work_dir: Path, out_dir: Path, medres_module, overwrite=False):
    local_zip = work_dir / Path(zip_url).name
    if not local_zip.exists() or overwrite:
        print(f"[GET ] {zip_url}")
        with requests.get(zip_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(local_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=1<<20):
                    if chunk:
                        f.write(chunk)
    else:
        print(f"[SKIP] already downloaded: {local_zip.name}")

    with zipfile.ZipFile(local_zip, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".fits")]
        print(f"[ZIP ] {local_zip.name}: {len(members)} FITS")
        for m in members:
            temp_fits = work_dir / Path(m).name
            with zf.open(m) as src, open(temp_fits, "wb") as dst:
                dst.write(src.read())
            try:
                out_path = convert_with_medres(medres_module, temp_fits, out_dir)
                print(f"[OK  ] {out_path.name}")
            except Exception as e:
                print(f"[FAIL] {temp_fits.name}: {e}", file=sys.stderr)
            finally:
                temp_fits.unlink(missing_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Fetch PHOENIX MedRes R=10000 spectra and convert with openmedres.py.")
    ap.add_argument("--base-url", default=BASE_URL, help="Base directory for R=10000 ZIPs.")
    ap.add_argument("--outdir", default="phoenix_out", help="Destination folder for .txt files.")
    ap.add_argument("--workdir", default="tmp", help="Temporary folder for downloads/extraction.")
    ap.add_argument("--overwrite", action="store_true", help="Re-download and re-convert even if files exist.")
    ap.add_argument("--medres-module", default="openmedres", help="Converter module name (default medres.py).")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.workdir)
    work_dir.mkdir(parents=True, exist_ok=True)

    zip_urls = list_all_zip_urls(args.base_url)
    if not zip_urls:
        print("No ZIPs found — check internet or the base URL.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(zip_urls)} ZIP archives.")
    for u in zip_urls:
        try:
            download_and_process_zip(u, work_dir, out_dir, medres_module, overwrite=args.overwrite)
        except Exception as e:
            print(f"[FAIL] {u}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
