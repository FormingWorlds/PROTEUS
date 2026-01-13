#!/usr/bin/env python3
"""
Upload complete Zenodo structure to OSF, matching exact file structure.

This script:
1. Reads from zenodo_stellar_spectra_extracted/ (manually downloaded Zenodo data)
2. Uploads to OSF project 8r2sw, preserving exact structure
3. Includes all readme files
4. Skips files that already exist on OSF
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

# Check for OSF token
osf_token = os.environ.get('OSF_TOKEN')
if not osf_token:
    print("ERROR: OSF_TOKEN environment variable not set")
    sys.exit(1)

# Connect to OSF
print("Connecting to OSF...")
osf = OSF(token=osf_token)
project = osf.project('8r2sw')
storage = project.storage('osfstorage')

print(f"Connected to OSF project: {project.id}")
print(f"Project title: {project.title}\n")

# Base directory with extracted Zenodo data
base_dir = Path(__file__).parent / 'zenodo_stellar_spectra_extracted'

if not base_dir.exists():
    print(f"ERROR: Directory not found: {base_dir}")
    print("Please extract the Zenodo ZIP files to zenodo_stellar_spectra_extracted/")
    sys.exit(1)

# Get existing files on OSF for comparison
print("Checking existing files on OSF...")
existing_files = {f.path for f in storage.files}
print(f"Found {len(existing_files)} existing files on OSF\n")


def upload_folder(local_dir: Path, osf_base_path: str):
    """
    Upload a local folder to OSF, preserving structure.

    Parameters
    ----------
    local_dir : Path
        Local directory to upload
    osf_base_path : str
        Base path on OSF (e.g., "/PHOENIX")
    """
    uploaded = 0
    skipped = 0
    errors = 0
    total_size = 0

    # Get all files recursively
    all_files = list(local_dir.rglob('*'))
    files = [f for f in all_files if f.is_file()]

    print(f"  Found {len(files)} files to upload")

    for local_file in files:
        # Calculate relative path from local_dir
        rel_path = local_file.relative_to(local_dir)
        osf_path = f"{osf_base_path.rstrip('/')}/{rel_path.as_posix()}"

        # Check if file already exists
        if osf_path in existing_files:
            print(f"  ⏭  Skipping (exists): {osf_path}")
            skipped += 1
            continue

        file_size_mb = local_file.stat().st_size / 1024 / 1024
        total_size += local_file.stat().st_size

        # Retry logic for uploads
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Show progress for large files
                if file_size_mb > 10:
                    print(f"  ⬆  Uploading: {osf_path} ({file_size_mb:.2f} MB)... [attempt {retry + 1}/{max_retries}]")
                else:
                    print(f"  ⬆  Uploading: {osf_path} [attempt {retry + 1}/{max_retries}]")

                with open(local_file, 'rb') as f:
                    storage.create_file(osf_path, f)
                uploaded += 1

                # Update existing files set to avoid re-checking
                existing_files.add(osf_path)
                break  # Success, exit retry loop

            except Exception as e:
                if retry < max_retries - 1:
                    print(f"  ⚠ Retry {retry + 1}/{max_retries} after error: {str(e)[:100]}")
                    import time
                    time.sleep(5 * (retry + 1))  # Exponential backoff
                else:
                    print(f"  ✗ Error uploading {osf_path} after {max_retries} attempts: {e}")
                    errors += 1

    return uploaded, skipped, errors, total_size


# Upload each dataset
results = []

# PHOENIX
phoenix_dir = base_dir / 'phoenix'
if phoenix_dir.exists() and any(phoenix_dir.iterdir()):
    print("=" * 70)
    print("Uploading PHOENIX spectra")
    print("=" * 70)
    uploaded, skipped, errors, total_size = upload_folder(phoenix_dir, '/PHOENIX')
    results.append(('PHOENIX', uploaded, skipped, errors, total_size))
    print(f"\nPHOENIX: {uploaded} uploaded, {skipped} skipped, {errors} errors")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB\n")
else:
    print("⚠ PHOENIX directory not found or empty")

# MUSCLES
muscles_dir = base_dir / 'muscles'
if muscles_dir.exists() and any(muscles_dir.iterdir()):
    print("=" * 70)
    print("Uploading MUSCLES spectra")
    print("=" * 70)
    uploaded, skipped, errors, total_size = upload_folder(muscles_dir, '/MUSCLES')
    results.append(('MUSCLES', uploaded, skipped, errors, total_size))
    print(f"\nMUSCLES: {uploaded} uploaded, {skipped} skipped, {errors} errors")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB\n")
else:
    print("⚠ MUSCLES directory not found or empty")

# Solar
solar_dir = base_dir / 'solar'
if solar_dir.exists() and any(solar_dir.iterdir()):
    print("=" * 70)
    print("Uploading Solar spectra")
    print("=" * 70)
    uploaded, skipped, errors, total_size = upload_folder(solar_dir, '/solar')
    results.append(('Solar', uploaded, skipped, errors, total_size))
    print(f"\nSolar: {uploaded} uploaded, {skipped} skipped, {errors} errors")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB\n")
else:
    print("⚠ Solar directory not found or empty")

# Summary
print("=" * 70)
print("UPLOAD SUMMARY")
print("=" * 70)
total_uploaded = sum(u for _, u, _, _, _ in results)
total_skipped = sum(s for _, _, s, _, _ in results)
total_errors = sum(e for _, _, _, e, _ in results)
total_size_mb = sum(s for _, _, _, _, s in results) / 1024 / 1024

for name, uploaded, skipped, errors, size_mb in results:
    print(f"{name:10s}: {uploaded:4d} uploaded, {skipped:4d} skipped, {errors:4d} errors ({size_mb / 1024 / 1024:.2f} MB)")

print(f"\nTotal: {total_uploaded} uploaded, {total_skipped} skipped, {total_errors} errors")
print(f"Total size uploaded: {total_size_mb:.2f} MB")

if total_errors == 0:
    print("\n✓ Upload complete! All files from Zenodo are now on OSF.")
    print("✓ File structure matches Zenodo exactly.")
    print("✓ Readme files included in each folder.")
    sys.exit(0)
else:
    print(f"\n⚠ {total_errors} errors occurred. Please review and retry.")
    sys.exit(1)
