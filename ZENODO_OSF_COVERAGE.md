# Zenodo-OSF Coverage Analysis

**Date**: 2026-01-13  
**Zenodo Community**: [PROTEUS Framework for Planetary Evolution](https://zenodo.org/communities/proteus_framework/)  
**OSF Archive**: [https://osf.io/8dumn](https://osf.io/8dumn)

## Summary

**Total Zenodo records in codebase**: 27  
**Records with OSF fallback**: 24 (89%)  
**Records missing from OSF**: 3 (11%)

## Zenodo Records Missing from OSF Archive

The following 3 Zenodo records are used in the PROTEUS codebase but are **NOT** in the OSF archive and do **NOT** have OSF fallback configured:

### 1. Zenodo Record 17674612 - PHOENIX Spectra
- **Function**: `download_phoenix()`
- **Purpose**: PHOENIX stellar spectra ZIP files
- **Location in code**: `src/proteus/utils/data.py:784`
- **Status**: ❌ No OSF fallback
- **Zenodo URL**: https://zenodo.org/record/17674612

### 2. Zenodo Record 17802209 - MUSCLES Spectra
- **Function**: `download_muscles()`
- **Purpose**: MUSCLES stellar spectra files
- **Location in code**: `src/proteus/utils/data.py:834`
- **Status**: ❌ No OSF fallback
- **Zenodo URL**: https://zenodo.org/record/17802209

### 3. Zenodo Record 17981836 - Solar Spectra
- **Function**: `download_all_solar_spectra()`
- **Purpose**: All solar spectra (nrel, VPL past, VPL present, VPL future)
- **Location in code**: `src/proteus/utils/data.py:772`
- **Status**: ❌ No OSF fallback
- **Zenodo URL**: https://zenodo.org/record/17981836

## Zenodo Records WITH OSF Fallback

All other 24 Zenodo records have OSF fallback configured in `DATA_SOURCE_MAP`:

### Spectral Files (OSF: vehxg)
- 15696415, 15696457, 15721749, 15743843
- 15799318, 15799474, 15799495
- 15799607, 15799652, 15799731
- 15799743, 15799754, 15799776

### Interior Lookup Tables (OSF: phsxf)
- 15728072, 15728091, 15728138
- 15877374, 15877424, 17417017

### Other Data (Various OSF projects)
- 15721440 (Stellar spectra, OSF: 8r2sw)
- 15727878 (Exoplanet data, OSF: fzwr4)
- 15727899 (Mass-radius data, OSF: xge8t)
- 15727998 (Population/EOS data, OSF: dpkjb)
- 15880455 (Surface albedos, OSF: 2gcd9)

## Recommendations

To improve robustness and ensure all data has fallback options:

1. **Add missing records to OSF archive**:
   - Upload PHOENIX spectra (17674612) to OSF
   - Upload MUSCLES spectra (17802209) to OSF
   - Upload Solar spectra (17981836) to OSF

2. **Update DATA_SOURCE_MAP** in `src/proteus/utils/data.py`:
   ```python
   'PHOENIX': {'zenodo_id': '17674612', 'osf_id': '<osf_id>', 'osf_project': '<osf_id>'},
   'MUSCLES': {'zenodo_id': '17802209', 'osf_id': '<osf_id>', 'osf_project': '<osf_id>'},
   'Solar': {'zenodo_id': '17981836', 'osf_id': '<osf_id>', 'osf_project': '<osf_id>'},
   ```

3. **Update download functions** to use mapping:
   - `download_phoenix()` - use mapping lookup
   - `download_muscles()` - use mapping lookup
   - `download_all_solar_spectra()` - use mapping lookup

## Impact

Currently, if Zenodo fails for these 3 records, the system cannot fall back to OSF, which could cause:
- Download failures in CI/CD pipelines
- User frustration when Zenodo is unavailable
- Reduced robustness compared to other data sources

Adding OSF fallback for these records would bring coverage to **100%** and ensure all data downloads have redundancy.
