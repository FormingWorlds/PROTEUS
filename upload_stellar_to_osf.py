#!/usr/bin/env python3
"""
Upload stellar spectral data to OSF project 8r2sw.

This script uploads:
- PHOENIX spectra (from Zenodo 17674612)
- MUSCLES spectra (from Zenodo 17802209)
- Solar spectra (from Zenodo 17981836)

Requirements:
- OSF API token in environment variable OSF_TOKEN
- Data already downloaded to FWL_DATA/stellar_spectra/
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from osfclient.api import OSF
except ImportError:
    print("ERROR: osfclient not installed. Install with: pip install osfclient")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from proteus.utils.data import GetFWLData


def upload_folder_to_osf(storage, local_path: Path, osf_base_path: str = "/"):
    """
    Upload a local folder to OSF storage.

    Parameters
    ----------
    storage : OSF storage object
        OSF storage to upload to
    local_path : Path
        Local directory to upload
    osf_base_path : str
        Base path in OSF (e.g., "/PHOENIX")
    """
    uploaded = 0
    skipped = 0
    errors = 0

    # Get all files recursively
    all_files = list(local_path.rglob('*'))
    files = [f for f in all_files if f.is_file()]

    print(f"  Found {len(files)} files to upload")

    for local_file in files:
        # Calculate relative path from local_path
        rel_path = local_file.relative_to(local_path)
        osf_path = f"{osf_base_path.rstrip('/')}/{rel_path.as_posix()}"

        # Check if file already exists
        existing = [f for f in storage.files if f.path == osf_path]
        if existing:
            print(f"  ⏭  Skipping (exists): {osf_path}")
            skipped += 1
            continue

        try:
            print(f"  ⬆  Uploading: {osf_path} ({local_file.stat().st_size / 1024 / 1024:.2f} MB)")
            with open(local_file, 'rb') as f:
                storage.create_file(osf_path, f)
            uploaded += 1
        except Exception as e:
            print(f"  ✗ Error uploading {osf_path}: {e}")
            errors += 1

    return uploaded, skipped, errors


def main():
    """Upload stellar spectral data to OSF."""
    # Check for OSF token
    osf_token = os.environ.get('OSF_TOKEN')
    if not osf_token:
        print("=" * 70)
        print("ERROR: OSF_TOKEN environment variable not set")
        print("=" * 70)
        print("\nTo get your OSF token:")
        print("1. Go to https://osf.io/settings/tokens/")
        print("2. Create a new token with 'osf.full_write' scope")
        print("3. Export it: export OSF_TOKEN='your_token_here'")
        print("\nThen run this script again.")
        return 1

    # Connect to OSF
    print("Connecting to OSF...")
    osf = OSF(token=osf_token)
    project = osf.project('8r2sw')
    storage = project.storage('osfstorage')

    print(f"Connected to OSF project: {project.id}")
    print(f"Project title: {project.title}\n")

    # Get local data directory
    fwl_data = GetFWLData()
    stellar_dir = fwl_data / 'stellar_spectra'

    if not stellar_dir.exists():
        print(f"ERROR: Stellar spectra directory not found: {stellar_dir}")
        print("Please download the data first using download_missing_stellar_data.py")
        return 1

    results = []

    # Upload PHOENIX
    phoenix_dir = stellar_dir / 'PHOENIX'
    if phoenix_dir.exists() and any(phoenix_dir.iterdir()):
        print("=" * 70)
        print("Uploading PHOENIX spectra")
        print("=" * 70)
        uploaded, skipped, errors = upload_folder_to_osf(storage, phoenix_dir, '/PHOENIX')
        results.append(('PHOENIX', uploaded, skipped, errors))
        print(f"\nPHOENIX: {uploaded} uploaded, {skipped} skipped, {errors} errors\n")
    else:
        print("⚠ PHOENIX directory not found or empty, skipping")

    # Upload MUSCLES
    muscles_dir = stellar_dir / 'MUSCLES'
    if muscles_dir.exists() and any(muscles_dir.iterdir()):
        print("=" * 70)
        print("Uploading MUSCLES spectra")
        print("=" * 70)
        uploaded, skipped, errors = upload_folder_to_osf(storage, muscles_dir, '/MUSCLES')
        results.append(('MUSCLES', uploaded, skipped, errors))
        print(f"\nMUSCLES: {uploaded} uploaded, {skipped} skipped, {errors} errors\n")
    else:
        print("⚠ MUSCLES directory not found or empty, skipping")

    # Upload Solar
    solar_dir = stellar_dir / 'solar'
    if solar_dir.exists() and any(solar_dir.iterdir()):
        print("=" * 70)
        print("Uploading Solar spectra")
        print("=" * 70)
        uploaded, skipped, errors = upload_folder_to_osf(storage, solar_dir, '/solar')
        results.append(('Solar', uploaded, skipped, errors))
        print(f"\nSolar: {uploaded} uploaded, {skipped} skipped, {errors} errors\n")
    else:
        print("⚠ Solar directory not found or empty, skipping")

    # Summary
    print("=" * 70)
    print("UPLOAD SUMMARY")
    print("=" * 70)
    total_uploaded = sum(u for _, u, _, _ in results)
    total_skipped = sum(s for _, _, s, _ in results)
    total_errors = sum(e for _, _, _, e in results)

    for name, uploaded, skipped, errors in results:
        print(f"{name:10s}: {uploaded:4d} uploaded, {skipped:4d} skipped, {errors:4d} errors")

    print(f"\nTotal: {total_uploaded} uploaded, {total_skipped} skipped, {total_errors} errors")

    if total_errors == 0:
        print("\n✓ Upload complete! You can now update the mapping in data.py")
        return 0
    else:
        print(f"\n⚠ {total_errors} errors occurred. Please review and retry.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
