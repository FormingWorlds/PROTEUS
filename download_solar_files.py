#!/usr/bin/env python3
"""Download Solar spectra files individually from Zenodo."""

from __future__ import annotations

import json
import subprocess as sp
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from proteus.utils.data import GetFWLData

zenodo_id = '17981836'
api_url = f"https://zenodo.org/api/records/{zenodo_id}"

solar_dir = GetFWLData() / 'stellar_spectra' / 'solar'
solar_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Solar spectra files...")
with urllib.request.urlopen(api_url, timeout=30) as response:
    data = json.loads(response.read())

files = data.get('files', [])
print(f"Found {len(files)} files\n")

for f in files:
    file_key = f.get('key', 'unknown')
    file_url = f['links']['self']
    file_path = solar_dir / file_key

    if file_path.exists():
        print(f"⏭  Skipping (exists): {file_key}")
        continue

    print(f"⬇  Downloading: {file_key}...")
    result = sp.run(
        ['curl', '-L', '-o', str(file_path), file_url],
        timeout=300,
        capture_output=True,
        text=True
    )

    if result.returncode == 0 and file_path.exists():
        size = file_path.stat().st_size / 1024 / 1024
        print(f"   ✓ {size:.2f} MB\n")
    else:
        print("   ✗ Failed\n")

file_count = sum(1 for f in solar_dir.iterdir() if f.is_file())
print(f"✓ Downloaded {file_count} files to {solar_dir}")
