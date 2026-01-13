# OSF Upload Status

**Date**: 2026-01-13  
**OSF Project**: https://osf.io/8r2sw  
**Token**: Provided and configured

## Current Status

### Data Download Status ✅

All data has been successfully downloaded from Zenodo:

- **PHOENIX**: 495 files, 411.90 MB ✅
- **MUSCLES**: 34 files, 781.34 MB ✅
- **Solar**: 10 files, 7.65 MB ✅

**Total**: 539 files, ~1.2 GB

### OSF Upload Status ⏳

Upload is in progress (running in background):

- **PHOENIX**: 34/495 files uploaded (28.29 MB) - **In Progress**
- **MUSCLES**: 0/34 files uploaded - **Pending**
- **Solar**: 0/10 files uploaded - **Pending**

## Upload Process

The upload script (`upload_stellar_to_osf.py`) is running in the background and will:

1. Upload all PHOENIX files to `/PHOENIX/` on OSF
2. Upload all MUSCLES files to `/MUSCLES/` on OSF
3. Upload all Solar files to `/solar/` on OSF

**Note**: Large file uploads (especially MUSCLES at 781 MB) will take time.

## Monitoring

To check upload progress:

```bash
export OSF_TOKEN='your_token'
python -c "
from proteus.utils.data import get_osf
storage = get_osf('8r2sw')
for folder in ['PHOENIX', 'MUSCLES', 'solar']:
    files = [f for f in storage.files if f.path.startswith(f'/{folder}/')]
    print(f'/{folder}/: {len(files)} files')
"
```

Or check directly on OSF: https://osf.io/8r2sw/files/osfstorage/

## Expected Final Structure

After upload completes, OSF project `8r2sw` will have:

```
/osfstorage/
  /Named/              (existing - 11 files)
  /BTSETTL-CIFIST/     (existing - 1 file)
  /PHOENIX/            (NEW - 495 files)
  /MUSCLES/            (NEW - 34 files)
  /solar/              (NEW - 10 files)
```

## Next Steps

Once upload completes:

1. ✅ Code mapping already added to `DATA_SOURCE_MAP`
2. ✅ All three datasets mapped to OSF project `8r2sw`
3. ✅ System will automatically use OSF fallback when Zenodo fails

**Result**: 100% OSF-Zenodo mirror coverage for all stellar spectral data!
