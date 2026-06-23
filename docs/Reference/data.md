# Reference data

This page documents the reference datasets that PROTEUS uses at runtime:
where they come from, what they contain, and how they are identified in
config files. For instructions on switching between spectrum sources, see
[Switch and choose a stellar spectrum](../How-to/switch_spectrum.md).
## Automatic data download

PROTEUS downloads reference data from
[Zenodo](https://zenodo.org/communities/proteus_framework/) on first run.
If Zenodo is unavailable, each dataset falls back automatically to its
corresponding project on the [Open Science Framework](https://osf.io/)
(OSF). A Zenodo API token is optional; without one, public access is used
with lower rate limits.

| Dataset | Downloaded by |
|---|---|
| Stellar spectra (solar, MUSCLES) | `proteus get solar`, `proteus get muscles` |
| PHOENIX synthetic spectra | `proteus get phoenix` |
| Stellar evolution tracks | `proteus get stellar` |
| Spectral k-tables | `proteus get spectral` |
| Surface albedos | `proteus get surfaces` |
| Scattering properties | `proteus get scattering` |
| Exoplanet populations, mass-radius curves | `proteus get reference` |
| Interior EOS tables, melting curves | `proteus get interiordata` |

To configure a Zenodo API token, see the
[Troubleshooting guide](../How-to/troubleshooting.md#data-download-errors-or-slow-zenodo-downloads).

---

## Stellar spectra

The spectrum source is set via `star.mors.spectrum_source` and
`star.mors.star_name` (or `star.mors.star_path` for a custom file).
See [Switch and choose a stellar spectrum](switch_stellar_spectrum.md)
for full usage instructions.

### Solar spectra

Observed solar spectra are stored under `$FWL_DATA/stellar_spectra/solar/`.
The modern spectrum is from [Gueymard (2003)](https://www.sciencedirect.com/science/article/pii/S0038092X03003967)
via [NREL](https://www.nrel.gov/grid/solar-resource/spectra.html).
Historical and future spectra are from
[Claire et al. (2012)](https://iopscience.iop.org/article/10.1088/0004-637X/757/1/95).

| `star_name` | Description | Age of Sun |
|---|---|---|
| `sun` | Modern NREL spectrum | 4.6 Gyr |
| `SunModern` | Alternative modern solar spectrum | 4.6 Gyr |
| `Sun0.6Ga` | 0.6 Gyr ago | ~4.0 Gyr |
| `Sun1.8Ga` | 1.8 Gyr ago | ~2.8 Gyr |
| `Sun2.4Ga` | 2.4 Gyr ago | ~2.2 Gyr |
| `Sun2.7Ga` | 2.7 Gyr ago | ~1.9 Gyr |
| `Sun3.8Ga` | 3.8 Gyr ago | ~800 Myr |
| `Sun4.4Ga` | 4.4 Gyr ago | ~200 Myr |
| `Sun5.6Gyr` | Future Sun | 5.6 Gyr |

`star_name` matching is case-insensitive.

### MUSCLES / Mega-MUSCLES spectra

Observed UV–optical–IR spectra from the
[MUSCLES](https://archive.stsci.edu/prepds/muscles/) and
[Mega-MUSCLES](https://archive.stsci.edu/prepds/mega-muscles/) surveys,
stored under `$FWL_DATA/stellar_spectra/MUSCLES/` as `<star_name>.txt`.
`star_name` matching is case-insensitive.

??? info "Full star catalog"

    | Star | `star_name` | Type | Teff (K) | Age | L (L☉) | M (M☉) | R (R☉) | Survey |
    |---|---|---|---|---|---|---|---|---|
    | [Epsilon Eridani](https://exoplanetarchive.ipac.caltech.edu/overview/eps%20Eri) | `v-eps-eri` | K2V | 5020 | 400–800 Myr | 0.32 | 0.82 | 0.759 | MUSCLES |
    | [GJ 1132](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%201132) | `gj1132` | M4.5V | 3229 | — | 0.005 | 0.195 | 0.221 | Mega-MUSCLES |
    | [GJ 1214](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%201214) | `gj1214` | M4V | 3100 | 5–10 Gyr | 0.00351 | 0.182 | 0.216 | MUSCLES |
    | [GJ 15 A](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%2015) | `gj15a` | M1–M2V | ~3700 | — | ~0.021 | ~0.40 | ~0.38 | Mega-MUSCLES |
    | [GJ 163](https://exoplanetarchive.ipac.caltech.edu/overview/gj%20163%20b) | `gj163` | M3.5V | ~3300–3500 | ~2–10 Gyr | ~0.02 | ~0.40 | ~0.41 | Mega-MUSCLES |
    | [GJ 176](https://exoplanetarchive.ipac.caltech.edu/overview/HD%20285968) | `gj176` | M2.5V | ~3700 | ~4 Gyr | 0.034 | 0.51 | 0.48 | MUSCLES |
    | [GJ 436](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20436) | `gj436` | M2.5–M3V | ~3600 | ~6–15 Gyr | 0.023 | 0.44 | 0.42 | MUSCLES |
    | [GJ 551 (Proxima Cen)](https://exoplanetarchive.ipac.caltech.edu/overview/alpha%20Cen) | `gj551` | M5.5Ve | ~2900–3000 | ~4.8 Gyr | 0.0015 | 0.12 | 0.14 | MUSCLES |
    | [GJ 581](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20581) | `gj581` | M3V | ~3500 | ~4 Gyr | ~0.012 | ~0.30 | ~0.30 | MUSCLES |
    | [GJ 649](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20649) | `gj649` | M1–M2V | ~3700 | — | ~0.044 | 0.51 | 0.50 | Mega-MUSCLES |
    | [GJ 667 C](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20667C) | `gj667c` | M1.5V | ~3700 | >2 Gyr | ~0.014 | ~0.33 | ~0.32–0.42 | MUSCLES |
    | [GJ 674](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20674) | `gj674` | M2.5V | ~3400–3600 | ~0.5–3 Gyr | 0.017 | 0.35 | 0.36 | Mega-MUSCLES |
    | [GJ 676 A](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20676A) | `gj676a` | M0V | ~3800–4000 | — | 0.083 | 0.63 | 0.65 | Mega-MUSCLES |
    | [GJ 699 (Barnard's Star)](https://exoplanetarchive.ipac.caltech.edu/overview/barnard's%20star) | `gj699` | M3.5–4V | ~3200–3300 | ~10 Gyr | 0.0035 | ~0.16 | ~0.19 | Mega-MUSCLES |
    | [GJ 729 (Ross 154)](https://simbad.cds.unistra.fr/simbad/sim-id?Ident=GJ+729) | `gj729` | M3.5V | ~3200–3300 | <1–2 Gyr | ~0.004–0.005 | ~0.18 | ~0.20 | Mega-MUSCLES |
    | [GJ 832](https://exoplanetarchive.ipac.caltech.edu/overview/HIP%20106440) | `gj832` | M1–M3V | ~3500 | ~4–12 Gyr | ~0.03 | 0.45 | 0.45 | MUSCLES |
    | GJ 832 (synthetic) | `gj832_synth` | M1–M3V | ~3500 | ~4–12 Gyr | ~0.03 | 0.45 | 0.45 | MUSCLES |
    | [GJ 849](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20849) | `gj849` | M3.5V | 3467 | >3 Gyr | 0.02887 | 0.45 | 0.45 | Mega-MUSCLES |
    | [GJ 876](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20876) | `gj876` | M2–4V | ~3200–3300 | ~2–15 Gyr | ~0.013 | ~0.37 | ~0.37 | MUSCLES |
    | [HAT-P-12](https://exoplanetarchive.ipac.caltech.edu/overview/HAT-P-12) | `hat-p-12` | K4V | 4500–4800 | 2–14 Gyr | ~0.20 | 0.74 | 0.70 | MUSCLES ext. |
    | [HAT-P-26](https://exoplanetarchive.ipac.caltech.edu/overview/HAT-P-26) | `hat-p-26` | K1V | ~5050 | 4–12 Gyr | ~0.44 | 0.85 | 0.86 | MUSCLES ext. |
    | [HD 40307](https://exoplanetarchive.ipac.caltech.edu/overview/HD%2040307) | `hd40307` | K2.5V | ~4800–5000 | ~2–5 Gyr | ~0.22 | ~0.79 | ~0.71 | MUSCLES |
    | [HD 85512](https://exoplanetarchive.ipac.caltech.edu/overview/HD%2085512) | `hd85512` | M0V | ~4400 | ~6 Gyr | 0.17 | 0.69 | 0.69 | MUSCLES |
    | [HD 97658](https://exoplanetarchive.ipac.caltech.edu/overview/HD%2097658%20b) | `hd97658` | K1V | 5212 | 3.9 Gyr | 0.351 | 0.773 | 0.728 | MUSCLES |
    | [HD 149026](https://exoplanetarchive.ipac.caltech.edu/overview/HD%20149026%20b) | `hd-149026` | G0V | ~6100–6200 | ~2–3 Gyr | ~2.6 | ~1.14 | ~1.46 | MUSCLES ext. |
    | [L 98-59](https://exoplanetarchive.ipac.caltech.edu/overview/L%2098-59) | `l-98-59` | M3V | 3415 | ~5 Gyr | 0.012 | 0.292 | 0.316 | Mega-MUSCLES |
    | [L 678-39 (GJ 357)](https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20357) | `l-678-39` | M2.5V | ~3500 | — | 0.0017 | ~0.34 | ~0.34 | MUSCLES ext. |
    | [L 980-5](https://simbad.u-strasbg.fr/simbad/sim-id?Ident=l+980-5) | `l-980-5` | M4V | — | — | — | — | — | Mega-MUSCLES |
    | [LHS 2686](https://simbad.cds.unistra.fr/simbad/sim-id?Ident=LHS+2686) | `lhs-2686` | M5V | — | — | — | — | — | Mega-MUSCLES |
    | [LP 791-18](https://exoplanetarchive.ipac.caltech.edu/overview/LP%20791-18) | `lp-791-18` | M6V | ~2960 | >0.5 Gyr | ~0.002 | 0.139 | 0.182 | MUSCLES ext. |
    | [TOI-193 (LTT 9779)](https://exoplanetarchive.ipac.caltech.edu/overview/LTT%209779) | `toi-193` | G7V | ~5400–5500 | ~2 Gyr | ~0.7 | ~1.0 | ~0.95 | MUSCLES ext. |
    | [TRAPPIST-1](https://exoplanetarchive.ipac.caltech.edu/overview/TRAPPIST-1) | `trappist-1` | M8V | 2566 | ~8 Gyr | 0.000553 | 0.0898 | 0.1192 | Mega-MUSCLES |
    | [WASP-17](https://exoplanetarchive.ipac.caltech.edu/overview/WASP-17) | `wasp-17` | F4–F6V | ~6550 | ~3 Gyr | ~4 | 1.35 | 1.57 | MUSCLES ext. |
    | [WASP-43](https://exoplanetarchive.ipac.caltech.edu/overview/WASP-43) | `wasp-43` | K7V | ~4100 | ~7 Gyr | ~0.15 | ~0.65 | ~0.76 | MUSCLES ext. |
    | [WASP-77 A](https://exoplanetarchive.ipac.caltech.edu/overview/WASP-77%20A) | `wasp-77a` | G8V | ~5600 | ~6 Gyr | ~0.74 | ~0.90 | ~0.91 | MUSCLES ext. |
    | [WASP-127](https://exoplanetarchive.ipac.caltech.edu/overview/WASP-127) | `wasp-127` | G5 | ~5600–5800 | ~10–12 Gyr | ~1.8 | ~0.95–1.10 | ~1.3 | MUSCLES ext. |

### PHOENIX synthetic spectra

Med-resolution synthetic spectra from the PHOENIX library, stored under
`$FWL_DATA/stellar_spectra/PHOENIX/<FeH>_<alpha>/`. Each subdirectory
corresponds to one metallicity–alpha combination (e.g. `FeH-0.5_alpha+0.0/`).

Parameters defining the PHOENIX grid:

| Parameter | Config key | Unit | Default |
|---|---|---|---|
| Effective temperature | `phoenix_Teff` | K | Derived from MORS tracks |
| Surface gravity | `phoenix_log_g` | log$_{10}$ (cgs) | Derived from MORS tracks |
| Stellar radius | `phoenix_radius` | R☉ | Derived from MORS tracks |
| Metallicity | `phoenix_FeH` | [Fe/H] dex | 0.0 (solar) |
| Alpha enhancement | `phoenix_alpha` | [α/Fe] dex | 0.0 (solar) |

Any parameter set to `"none"` in the config is derived from the MORS
stellar evolution tracks using `star.mass`. Grid inputs are automatically
rounded to the nearest available grid point.

### Custom stellar spectrum

A user-supplied spectrum is loaded when `star.mors.star_path` is set. It
overrides `spectrum_source`, `star_name`, and all `phoenix_*` parameters.
Environment variables (e.g. `$FWL_DATA`, `$HOME`) in the path are expanded
at load time.

**Required file format:**

| Column | Quantity | Unit |
|---|---|---|
| 1 | Wavelength | nm |
| 2 | Flux | erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$ |

The spectrum must be scaled to a distance of 1 AU. Comment lines begin
with `#`.

---

## Spectral files

Correlated-k opacity tables used by the atmosphere climate modules
(AGNI, JANUS). Selected via `atmos_clim.spectral_group` and
`atmos_clim.spectral_bands` in the config. Stored under
`$FWL_DATA/spectral_files/<group>/<bands>/`.
For a full description of each group's spectral coverage and gas
species, see `docs/assets/spectral_files.pdf` in the PROTEUS repository.

| Group | Available band counts | Notes |
|---|---|---|
| `Honeyside` | 16, 48, 256, 4096 | Default; broad gas coverage |
| `Dayspring` | 16, 48, 256, 4096 | |
| `Frostflow` | 16, 48, 256, 4096 | |
| `Oak` | 318 | |

The default configuration uses `Honeyside` / `48`. Higher band counts
increase spectral resolution and runtime cost.

---

## Surfaces

Single-scattering albedo data from Hammond et al. (2024):
[Zenodo record 13691960](https://zenodo.org/records/13691960).

Available surface types can be listed with:

```console
ls $FWL_DATA/surface_albedos/Hammond24
```

---

## Exoplanet population data

Obtained from the DACE PlanetS database
([Parc et al., 2024](https://arxiv.org/abs/2406.04311)).

---

## Mass-radius relations

Obtained from [Zeng et al. (2016)](https://iopscience.iop.org/article/10.3847/0004-637X/819/2/127/meta).