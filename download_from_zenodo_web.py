#!/usr/bin/env python3
"""
Download Zenodo records using web interface (bypasses API restrictions).

Uses direct download URLs from Zenodo web interface.
"""

from __future__ import annotations

import subprocess as sp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from proteus.utils.data import GetFWLData


def download_zenodo_web(zenodo_id: str, output_dir: Path) -> bool:
    """
    Download Zenodo record using web interface.

    Uses curl to download the record as a ZIP file.
    """
    print(f"  Downloading Zenodo {zenodo_id} via web interface...")

    # Get the actual file URLs from Zenodo API
    # First, try to get the record metadata to find download links
    import json
    import urllib.request

    try:
        api_url = f"https://zenodo.org/api/records/{zenodo_id}"
        with urllib.request.urlopen(api_url, timeout=30) as response:
            data = json.loads(response.read())

        # Find the ZIP file in the files list
        files = data.get('files', [])
        zip_file = None
        for f in files:
            if f.get('key', '').endswith('.zip') or 'zip' in f.get('type', '').lower():
                zip_file = f
                break

        if not zip_file:
            # Try to find any downloadable file
            zip_file = files[0] if files else None

        if zip_file:
            file_key = zip_file.get('key', 'file')
            print(f"  Found download URL: {file_key}")
        else:
            print("  ⚠ No downloadable files found")
            if not files:
                print("  ✗ No files found in Zenodo record")
                return False
    except Exception as e:
        print(f"  ⚠ Could not get metadata: {e}")
        if not files:
            print("  ✗ No files found in Zenodo record")
            return False

    # Download all files from the record
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download all files
        downloaded_files = 0
        for f in files:
            file_url = f['links']['self']
            file_key = f.get('key', 'unknown')
            file_size = f.get('size', 0) / 1024 / 1024  # MB

            # Determine output path
            if '/' in file_key:
                # Preserve directory structure
                file_path = output_dir / file_key
                file_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                file_path = output_dir / file_key

            # Skip if already exists
            if file_path.exists():
                print(f"  ⏭  Skipping (exists): {file_key}")
                continue

            print(f"  Downloading: {file_key} ({file_size:.2f} MB)...")
            result = sp.run(
                ['curl', '-L', '-o', str(file_path), file_url],
                timeout=600,
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and file_path.exists() and file_path.stat().st_size > 0:
                downloaded_files += 1
                print(f"    ✓ Downloaded {file_path.stat().st_size / 1024 / 1024:.2f} MB")
            else:
                print(f"    ✗ Failed: {result.stderr[:100]}")

        # Count total files
        all_files = list(output_dir.rglob('*'))
        file_count = sum(1 for f in all_files if f.is_file())
        print(f"  ✓ Total: {file_count} files in directory")

        return file_count > 0

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Download missing stellar data from Zenodo web interface."""
    print("=" * 70)
    print("Downloading Stellar Spectral Data from Zenodo (Web Interface)")
    print("=" * 70)

    fwl_data = GetFWLData()
    stellar_dir = fwl_data / 'stellar_spectra'
    stellar_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ('PHOENIX', '17674612', stellar_dir / 'PHOENIX'),
        ('MUSCLES', '17802209', stellar_dir / 'MUSCLES'),
        ('Solar', '17981836', stellar_dir / 'solar'),
    ]

    results = []
    for name, zenodo_id, output_dir in datasets:
        print(f"\n{name} (Zenodo {zenodo_id})")
        print("-" * 70)

        if output_dir.exists() and any(output_dir.iterdir()):
            print("  ⏭  Directory already exists with files, skipping")
            results.append((name, True))
            continue

        success = download_zenodo_web(zenodo_id, output_dir)
        results.append((name, success))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {name}")

    all_success = all(success for _, success in results)
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
