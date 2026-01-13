#!/usr/bin/env python3
"""
Download missing stellar spectral data from Zenodo for OSF mirroring.

Downloads:
- PHOENIX spectra (Zenodo 17674612)
- MUSCLES spectra (Zenodo 17802209)
- Solar spectra (Zenodo 17981836)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from proteus.utils.data import (
    GetFWLData,
    download_zenodo_folder,
    get_zenodo_file,
)


def download_phoenix_data():
    """Download PHOENIX spectra from Zenodo."""
    print("=" * 70)
    print("Downloading PHOENIX Spectra (Zenodo 17674612)")
    print("=" * 70)

    phoenix_zenodo_id = "17674612"
    data_dir = GetFWLData() / "stellar_spectra" / "PHOENIX"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download readme first to see structure
    print("Downloading readme...")
    get_zenodo_file(
        zenodo_id=phoenix_zenodo_id,
        folder_dir=data_dir,
        zenodo_path="_readme.md",
    )

    # Download entire folder structure
    print("Downloading PHOENIX folder structure...")
    result = download_zenodo_folder(
        zenodo_id=phoenix_zenodo_id,
        folder_dir=data_dir,
    )

    if result:
        files = list(data_dir.rglob('*'))
        file_count = sum(1 for f in files if f.is_file())
        print(f"✓ Downloaded {file_count} files from PHOENIX")
        return True
    else:
        print("✗ Failed to download PHOENIX data")
        return False


def download_muscles_data():
    """Download MUSCLES spectra from Zenodo."""
    print("\n" + "=" * 70)
    print("Downloading MUSCLES Spectra (Zenodo 17802209)")
    print("=" * 70)

    muscles_zenodo_id = "17802209"
    data_dir = GetFWLData() / "stellar_spectra" / "MUSCLES"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download readme first
    print("Downloading readme...")
    get_zenodo_file(
        zenodo_id=muscles_zenodo_id,
        folder_dir=data_dir,
        zenodo_path="_readme.md",
    )

    # Download entire folder
    print("Downloading MUSCLES folder structure...")
    result = download_zenodo_folder(
        zenodo_id=muscles_zenodo_id,
        folder_dir=data_dir,
    )

    if result:
        files = list(data_dir.rglob('*'))
        file_count = sum(1 for f in files if f.is_file())
        print(f"✓ Downloaded {file_count} files from MUSCLES")
        return True
    else:
        print("✗ Failed to download MUSCLES data")
        return False


def download_solar_data():
    """Download Solar spectra from Zenodo."""
    print("\n" + "=" * 70)
    print("Downloading Solar Spectra (Zenodo 17981836)")
    print("=" * 70)

    solar_zenodo_id = "17981836"
    data_dir = GetFWLData() / "stellar_spectra" / "solar"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading solar spectra folder...")
    result = download_zenodo_folder(
        zenodo_id=solar_zenodo_id,
        folder_dir=data_dir,
    )

    if result:
        files = list(data_dir.rglob('*'))
        file_count = sum(1 for f in files if f.is_file())
        print(f"✓ Downloaded {file_count} files from Solar spectra")
        return True
    else:
        print("✗ Failed to download Solar spectra")
        return False


def main():
    """Download all missing stellar spectral data."""
    print("Downloading missing stellar spectral data from Zenodo")
    print(f"Target directory: {GetFWLData() / 'stellar_spectra'}\n")

    results = []
    results.append(("PHOENIX", download_phoenix_data()))
    results.append(("MUSCLES", download_muscles_data()))
    results.append(("Solar", download_solar_data()))

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
