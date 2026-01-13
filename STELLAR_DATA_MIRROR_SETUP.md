# Stellar Spectral Data OSF-Zenodo Mirror Setup

**Date**: 2026-01-13
**OSF Project**: https://osf.io/8r2sw
**Goal**: Complete 1-to-1 mirror of stellar spectral data from Zenodo to OSF

## Overview

Three stellar spectral datasets are currently missing from the OSF archive:
1. **PHOENIX spectra** (Zenodo 17674612)
2. **MUSCLES spectra** (Zenodo 17802209)
3. **Solar spectra** (Zenodo 17981836)

## Quick Start

### Step 1: Get OSF API Token

1. Go to https://osf.io/settings/tokens/
2. Click "Create token"
3. Name: "PROTEUS Data Upload"
4. Scopes: Select `osf.full_write`
5. Copy the token

### Step 2: Set Environment Variable

```bash
export OSF_TOKEN='your_token_here'
```

### Step 3: Download Data from Zenodo (if needed)

```bash
python download_missing_stellar_data.py
```

This will download:
- PHOENIX → `$FWL_DATA/stellar_spectra/PHOENIX/`
- MUSCLES → `$FWL_DATA/stellar_spectra/MUSCLES/`
- Solar → `$FWL_DATA/stellar_spectra/solar/`

### Step 4: Upload to OSF

```bash
python upload_stellar_to_osf.py
```

This will upload:
- PHOENIX → `/PHOENIX/` on OSF
- MUSCLES → `/MUSCLES/` on OSF
- Solar → `/solar/` on OSF

### Step 5: Verify Upload

Check: https://osf.io/8r2sw/files/osfstorage/

You should see:
- `/PHOENIX/` folder
- `/MUSCLES/` folder
- `/solar/` folder

### Step 6: Code Mapping (Already Done!)

The mapping has been added to `src/proteus/utils/data.py`:

```python
'PHOENIX': {'zenodo_id': '17674612', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
'MUSCLES': {'zenodo_id': '17802209', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
'Solar': {'zenodo_id': '17981836', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
```

## Files Created

1. **`download_missing_stellar_data.py`** - Downloads data from Zenodo
2. **`upload_stellar_to_osf.py`** - Uploads data to OSF
3. **`OSF_UPLOAD_INSTRUCTIONS.md`** - Detailed instructions
4. **`update_mapping_after_upload.patch`** - Code mapping patch (already applied)

## Expected OSF Structure

After upload, OSF project `8r2sw` will have:

```
/osfstorage/
  /Named/              (existing)
  /BTSETTL-CIFIST/     (existing)
  /PHOENIX/            (NEW)
    /_readme.md
    /FeH+0.5_alpha+0.0_phoenixMedRes_R05000/
      ...
  /MUSCLES/            (NEW)
    /_readme.md
    /trappist-1.txt
    /gj1132.txt
    ...
  /solar/              (NEW)
    /nrel.txt
    /vpl_past.txt
    /vpl_present.txt
    /vpl_future.txt
```

## Verification

After upload, test the mapping:

```python
from proteus.utils.data import get_data_source_info, get_osf_from_zenodo

# Check mappings
for name in ['PHOENIX', 'MUSCLES', 'Solar']:
    info = get_data_source_info(name)
    print(f"{name}: Zenodo={info['zenodo_id']}, OSF={info['osf_project']}")

# Verify reverse lookup
for zid in ['17674612', '17802209', '17981836']:
    osf = get_osf_from_zenodo(zid)
    print(f"Zenodo {zid} → OSF {osf}")
```

## Troubleshooting

See `OSF_UPLOAD_INSTRUCTIONS.md` for detailed troubleshooting.

## Status

- ✅ Code mapping added to `DATA_SOURCE_MAP`
- ⏳ Data download script ready
- ⏳ Data upload script ready
- ⏳ **Action needed**: Run upload script after getting OSF token

Once uploaded, the OSF-Zenodo mirror will be **100% complete** for all stellar spectral data!
