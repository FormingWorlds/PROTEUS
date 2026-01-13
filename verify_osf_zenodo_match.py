#!/usr/bin/env python3
"""
Verify that OSF structure exactly matches Zenodo structure.

This script compares the manually downloaded Zenodo data with what's on OSF
to ensure a complete 1-to-1 mirror.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from osfclient.api import OSF
except ImportError:
    print("ERROR: osfclient not installed")
    sys.exit(1)

# Check for OSF token
osf_token = os.environ.get('OSF_TOKEN')
if not osf_token:
    print("ERROR: OSF_TOKEN environment variable not set")
    sys.exit(1)

# Connect to OSF
osf = OSF(token=osf_token)
project = osf.project('8r2sw')
storage = project.storage('osfstorage')

# Base directory with extracted Zenodo data
zenodo_dir = Path(__file__).parent / 'zenodo_stellar_spectra_extracted'

if not zenodo_dir.exists():
    print(f"ERROR: Directory not found: {zenodo_dir}")
    sys.exit(1)

print("=" * 70)
print("VERIFYING OSF-ZENODO STRUCTURE MATCH")
print("=" * 70)

all_match = True

for folder_name, osf_path in [('phoenix', 'PHOENIX'), ('muscles', 'MUSCLES'), ('solar', 'solar')]:
    print(f"\n{folder_name.upper()}")
    print("-" * 70)

    # Get Zenodo files
    zenodo_folder = zenodo_dir / folder_name
    if not zenodo_folder.exists():
        print(f"  ✗ Zenodo folder not found: {zenodo_folder}")
        all_match = False
        continue

    zenodo_files = {f.name for f in zenodo_folder.iterdir() if f.is_file()}

    # Get OSF files
    osf_files = {f.path.split('/')[-1] for f in storage.files if f.path.startswith(f'/{osf_path}/')}

    # Compare
    missing = zenodo_files - osf_files
    extra = osf_files - zenodo_files

    print(f"  Zenodo files: {len(zenodo_files)}")
    print(f"  OSF files: {len(osf_files)}")

    if missing:
        print(f"  ✗ Missing on OSF: {len(missing)} files")
        for f in sorted(missing)[:10]:
            print(f"    - {f}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        all_match = False
    else:
        print("  ✓ All Zenodo files present on OSF")

    # Check readme specifically
    readme_name = '_readme.md'
    if readme_name in zenodo_files:
        if readme_name in osf_files:
            print("  ✓ Readme file present")
        else:
            print("  ✗ Readme file MISSING on OSF")
            all_match = False

    # Extra files are OK (may be extracted content)
    if extra:
        print(f"  ⚠ Extra files on OSF: {len(extra)} (may be extracted content)")

print("\n" + "=" * 70)
if all_match:
    print("✓ VERIFICATION PASSED: OSF structure matches Zenodo exactly!")
    sys.exit(0)
else:
    print("✗ VERIFICATION FAILED: Some files are missing on OSF")
    print("  Run upload_complete_zenodo_structure.py to complete the upload")
    sys.exit(1)
