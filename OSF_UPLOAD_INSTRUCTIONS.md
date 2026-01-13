# OSF Upload Instructions for Stellar Spectral Data

This document provides step-by-step instructions for uploading the missing stellar spectral data to OSF project `8r2sw` to complete the Zenodo-OSF mirror.

## Missing Data to Upload

1. **PHOENIX spectra** (Zenodo 17674612) → `/PHOENIX/` on OSF
2. **MUSCLES spectra** (Zenodo 17802209) → `/MUSCLES/` on OSF
3. **Solar spectra** (Zenodo 17981836) → `/solar/` on OSF

## Prerequisites

1. **Python environment** with required packages:
   ```bash
   pip install osfclient
   ```

2. **OSF API Token**:
   - Go to https://osf.io/settings/tokens/
   - Click "Create token"
   - Name: "PROTEUS Data Upload"
   - Scopes: Select `osf.full_write`
   - Click "Create"
   - Copy the token (you'll only see it once!)

3. **Download the data from Zenodo** (if not already downloaded):
   ```bash
   python download_missing_stellar_data.py
   ```

## Step 1: Set OSF Token

Export the token as an environment variable:

```bash
export OSF_TOKEN='your_token_here'
```

Or add it to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export OSF_TOKEN="your_token_here"' >> ~/.zshrc
source ~/.zshrc
```

## Step 2: Run Upload Script

```bash
python upload_stellar_to_osf.py
```

The script will:
- Connect to OSF project `8r2sw`
- Upload PHOENIX data to `/PHOENIX/`
- Upload MUSCLES data to `/MUSCLES/`
- Upload Solar data to `/solar/`
- Skip files that already exist
- Show progress and summary

## Step 3: Verify Upload

Check the OSF project to verify files are uploaded:
- https://osf.io/8r2sw/files/osfstorage/

You should see:
- `/PHOENIX/` folder with PHOENIX spectra files
- `/MUSCLES/` folder with MUSCLES spectra files
- `/solar/` folder with solar spectra files

## Step 4: Update Code Mapping

Once uploaded, update `src/proteus/utils/data.py`:

1. Add to `DATA_SOURCE_MAP`:
   ```python
   # Stellar spectra - PHOENIX (OSF project: 8r2sw)
   'PHOENIX': {'zenodo_id': '17674612', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
   # Stellar spectra - MUSCLES (OSF project: 8r2sw)
   'MUSCLES': {'zenodo_id': '17802209', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
   # Stellar spectra - Solar (OSF project: 8r2sw)
   'Solar': {'zenodo_id': '17981836', 'osf_id': '8r2sw', 'osf_project': '8r2sw'},
   ```

2. Update download functions to use mapping (optional but recommended)

## Manual Upload Alternative

If the script doesn't work, you can upload manually via OSF web interface:

1. Go to https://osf.io/8r2sw/files/osfstorage/
2. Create folders: `PHOENIX`, `MUSCLES`, `solar`
3. Upload files from:
   - `$FWL_DATA/stellar_spectra/PHOENIX/` → `/PHOENIX/`
   - `$FWL_DATA/stellar_spectra/MUSCLES/` → `/MUSCLES/`
   - `$FWL_DATA/stellar_spectra/solar/` → `/solar/`

## Troubleshooting

### "OSF_TOKEN environment variable not set"
- Make sure you exported the token: `export OSF_TOKEN='your_token'`
- Check with: `echo $OSF_TOKEN`

### "osfclient not installed"
- Install: `pip install osfclient`

### "Permission denied" or "Forbidden"
- Check that your token has `osf.full_write` scope
- Verify you have write access to project `8r2sw`

### Upload fails for large files
- OSF has file size limits (typically 5GB per file)
- Very large files may need to be split or compressed
- Check OSF documentation for current limits

## Expected Folder Structure on OSF

After upload, OSF project `8r2sw` should have:

```
/osfstorage/
  /Named/              (already exists)
  /BTSETTL-CIFIST/     (already exists)
  /PHOENIX/            (NEW - from Zenodo 17674612)
    /_readme.md
    /FeH+0.5_alpha+0.0_phoenixMedRes_R05000/
      ...
  /MUSCLES/            (NEW - from Zenodo 17802209)
    /_readme.md
    /trappist-1.txt
    /gj1132.txt
    ...
  /solar/              (NEW - from Zenodo 17981836)
    /nrel.txt
    /vpl_past.txt
    /vpl_present.txt
    /vpl_future.txt
```
