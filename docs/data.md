# Reference data

## Contents

- [Stellar spectra](#stellar-spectra)
  - [Using solar spectrum](#using-solar-spectrum)
  - [Using a (Mega-)MUSCLES observed spectrum](#using-a-mega-muscles-observed-spectrum)
  - [Using a PHOENIX synthetic spectrum](#using-a-phoenix-synthetic-spectrum)
  - [Using a custom stellar spectrum](#using-a-custom-stellar-spectrum)
- [Surfaces](#surfaces)
- [Exoplanet population data](#exoplanet-population-data)
- [Mass-radius relations](#mass-radius-relations)

## Stellar spectra

PROTEUS can use:
- **Observed spectra** from MUSCLES / Mega-MUSCLES and NREL (for the Sun)
- **Synthetic PHOENIX spectra**

By default, observed spectra are searched in:

- `$FWL_DATA/stellar_spectra/MUSCLES`  – MUSCLES / Mega-MUSCLES stars
- `$FWL_DATA/stellar_spectra/solar`    – solar spectra (modern, past, future)

To see which files you have installed, run for example:

```console
ls $FWL_DATA/stellar_spectra/MUSCLES
ls $FWL_DATA/stellar_spectra/solar
```

For PHOENIX spectra, files are stored under: `$FWL_DATA/stellar_spectra/PHOENIX`, where each subdirectory corresponds to a metallicity / alpha combination, e.g. FeH-0.5_alpha+0.0/.

### Using solar spectra

To use the modern NREL observed spectrum of the Sun, set `spectrum_source = "solar"` or do not set the parameter at all, and set `star_name = "sun"`.

To use a different solar spectrum, for example a **'young sun'** spectrum, there are [VPL spectra](https://live-vpl-test.pantheonsite.io/models/evolution-of-solar-flux/) available by Claire et al. (2012). In this case, set `spectrum_source = "solar"` and choose one of the following options for `star_name`:

- `Sun0.6Ga` A young sun of 0.6 Gyr ago (age ~ 4.0 Gyr)
- `Sun1.8Ga` A young sun of 1.8 Gyr ago (age ~ 2.8 Gyr)
- `Sun2.4Ga` A young sun of 2.4 Gyr ago (age ~ 2.2 Gyr)
- `Sun2.7Ga` A young sun of 2.7 Gyr ago (age ~ 1.9 Gyr)
- `Sun3.8Ga` A young sun of 3.8 Gyr ago (age ~ 800 Myr)
- `Sun4.4Ga` A young sun of 4.4 Gyr ago (age ~ 200 Myr)
- `Sun5.6Gyr` A future sun of an age of 5.6 Gyr
- `SunModern` A modern solar spectrum, can be chosen as an alternative to the NREL spectrum listed above.


####  The Sun

* **Star name:**  `sun`
* **URL:** https://en.wikipedia.org/wiki/Sun
* **Spectral type:**  G2V
* **Teff:**  5772 K
* **Age:**   4.6 Gyr
* **Luminosity:**   1.0 L☉
* **Mass:**     1.0 M☉
* **Radius:**     1.0 R☉
* **Source:** [Gueymard 2003](https://www.sciencedirect.com/science/article/pii/S0038092X03003967), [NREL](https://www.nrel.gov/grid/solar-resource/spectra.html), [Claire et al. 2012](https://iopscience.iop.org/article/10.1088/0004-637X/757/1/95)

### Using a (Mega-)MUSCLES observed spectrum

PROTEUS can use observed stellar spectra from the MUSCLES and Mega-MUSCLES surveys. To use one of these observed spectra:

- Set `spectrum_source = "muscles"`
- Set `star_name` to one of the keys listed below (these names **must match** the installed spectrum filenames / dataset keys; many targets have multiple common aliases, but PROTEUS expects the specific `star_name` string shown here).

By default, PROTEUS looks for MUSCLES/Mega-MUSCLES spectra under:

- `$FWL_DATA/stellar_spectra/MUSCLES`

Files are saved as `<star_name>.txt`, e.g. `trappist-1.txt`.

#### Epsilon Eridani

- **Star name:** `v-eps-eri`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/eps%20Eri
- **Spectral type:** K2V
- **Teff:** 5020 K
- **Age:** 400–800 Myr
- **Luminosity:** 0.32 L☉
- **Mass:** 0.82 M☉
- **Radius:** 0.759 R☉
- **Source:** MUSCLES

---

#### GJ 1132

- **Star name:** `gj1132`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%201132
- **Spectral type:** M4.5 V
- **Teff:** 3229 K
- **Age:** ?
- **Luminosity:** 0.005 L☉
- **Mass:** 0.195 M☉
- **Radius:** 0.221 R☉
- **Source:** Mega-MUSCLES

---

#### GJ 1214

- **Star name:** `gj1214`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%201214
- **Spectral type:** M4 V
- **Teff:** 3100 K
- **Age:** 5–10 Gyr
- **Luminosity:** 0.00351 L☉
- **Mass:** 0.182 M☉
- **Radius:** 0.216 R☉
- **Source:** MUSCLES

---

#### HD 85512

- **Star name:** `hd85512`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/HD%2085512
- **Spectral type:** M0V
- **Teff:** ~4400 K
- **Age:** ~6 Gyr
- **Luminosity:** 0.17 L☉
- **Mass:** 0.69 M☉
- **Radius:** 0.69 R☉
- **Source:** MUSCLES

---

#### HD 97658

- **Star name:** `hd97658`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/HD%2097658%20b#planet_HD-97658-b_collapsible
- **Spectral type:** K1V
- **Teff:** 5212 K
- **Age:** 3.9 Gyr
- **Luminosity:** 0.351 L☉
- **Mass:** 0.773 M☉
- **Radius:** 0.728 R☉
- **Source:** MUSCLES

---

#### L 98-59

- **Star name:** `l-98-59`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/L%2098-59
- **Spectral type:** M3V
- **Teff:** 3415 K
- **Age:** ~5 Gyr
- **Luminosity:** 0.012 L☉
- **Mass:** 0.292 M☉
- **Radius:** 0.316 R☉
- **Source:** Mega-MUSCLES

---

#### TRAPPIST-1

- **Star name:** `trappist-1`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/TRAPPIST-1
- **Spectral type:** M8V
- **Teff:** 2566 K
- **Age:** ~8 Gyr
- **Luminosity:** 0.000553 L☉
- **Mass:** 0.0898 M☉
- **Radius:** 0.1192 R☉
- **Source:** Mega-MUSCLES

---

#### GJ 849

- **Star name:** `gj849`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20849
- **Spectral type:** M3.5V
- **Teff:** 3467 K
- **Age:** >3 Gyr
- **Luminosity:** 0.02887 L☉
- **Mass:** 0.45 M☉
- **Radius:** 0.45 R☉
- **Source:** Mega-MUSCLES

---

#### WASP-43

- **Star name:** `wasp-43`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/WASP-43
- **Spectral type:** K7V
- **Teff:** ~4100 K
- **Age:** ~7 Gyr
- **Luminosity:** ~0.15 L☉
- **Mass:** ~0.65 M☉
- **Radius:** ~0.76 R☉
- **Source:** MUSCLES extension

---

#### WASP-77 A

- **Star name:** `wasp-77a`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/WASP-77%20A
- **Spectral type:** G8V
- **Teff:** ~5600 K
- **Age:** ~6 Gyr
- **Luminosity:** ~0.74 L☉
- **Mass:** ~0.90 M☉
- **Radius:** ~0.91 R☉
- **Source:** MUSCLES extension

---

#### GJ 15 A

- **Star name:** `gj15a`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%2015
- **Spectral type:** M1–M2V
- **Teff:** ~3700 K
- **Age:** ?
- **Luminosity:** ~0.021 L☉
- **Mass:** ~0.40 M☉
- **Radius:** ~0.38 R☉
- **Source:** Mega-MUSCLES

---

#### GJ 163

- **Star name:** `gj163`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/gj%20163%20b
- **Spectral type:** M3.5V
- **Teff:** ~3300–3500 K
- **Age:** ~2-10 Gyr
- **Luminosity:** ~0.02 L☉
- **Mass:** ~0.40 M☉
- **Radius:** ~0.41 R☉
- **Source:** Mega-MUSCLES

---

#### GJ 176 (HD 285968)

- **Star name:** `gj176`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/HD%20285968
- **Spectral type:** M2.5V
- **Teff:** ~3700 K
- **Age:** ~4 Gyr
- **Luminosity:** 0.034 L☉
- **Mass:** 0.51 M☉
- **Radius:** 0.48 R☉
- **Source:** MUSCLES

---

#### GJ 436

- **Star name:** `gj436`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20436
- **Spectral type:** M2.5-M3V
- **Teff:** ~3600 K
- **Age:** ~6–15 Gyr
- **Luminosity:** 0.023 L☉
- **Mass:** 0.44 M☉
- **Radius:** 0.42 R☉
- **Source:** MUSCLES

---

#### GJ 551 (Proxima Centauri)

- **Star name:** `gj551`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/alpha%20Cen
- **Spectral type:** M5.5Ve
- **Teff:** ~2900-3000 K
- **Age:** ~4.8 Gyr
- **Luminosity:** 0.0015 L☉
- **Mass:** 0.12 M☉
- **Radius:** 0.14 R☉
- **Source:** MUSCLES

---

#### GJ 581

- **Star name:** `gj581`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20581
- **Spectral type:** M3V
- **Teff:** ~3500 K
- **Age:** ~4 Gyr
- **Luminosity:** ~0.012 L☉
- **Mass:** ~0.30 M☉
- **Radius:** ~0.30 R☉
- **Source:** MUSCLES

---

#### GJ 649

- **Star name:** `gj649`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20649
- **Spectral type:** M1-M2V
- **Teff:** ~3700 K
- **Age:** ?
- **Luminosity:** ~0.044 L☉
- **Mass:** 0.51 M☉
- **Radius:** 0.50 R☉
- **Source:** Mega-MUSCLES

---

#### GJ 667 C

- **Star name:** `gj667c`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20667C
- **Spectral type:** M1.5V
- **Teff:** ~3700 K
- **Age:** >2 Gyr
- **Luminosity:** ~0.014 L☉
- **Mass:** ~0.33 M☉
- **Radius:** ~0.32–0.42 R☉
- **Source:** MUSCLES

---

#### GJ 674

- **Star name:** `gj674`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20674
- **Spectral type:** M2.5V
- **Teff:** ~3400–3600 K
- **Age:** ~0.5–3 Gyr
- **Luminosity:** 0.017 L☉
- **Mass:** 0.35 M☉
- **Radius:** 0.36 R☉
- **Source:** Mega-MUSCLES

---

#### GJ 676 A

- **Star name:** `gj676a`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20676A
- **Spectral type:** M0V
- **Teff:** ~3800–4000 K
- **Age:** ?
- **Luminosity:** 0.083 L☉
- **Mass:** 0.63 M☉
- **Radius:** 0.65 R☉
- **Source:** Mega-MUSCLES

---

#### GJ 699 (Barnard’s Star)

- **Star name:** `gj699`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/barnard's%20star
- **Spectral type:** M3.5-4.0V
- **Teff:** ~3200-3300 K
- **Age:** ~10 Gyr
- **Luminosity:** 0.0035 L☉
- **Mass:** ~0.16 M☉
- **Radius:** ~0.19 R☉
- **Source:** Mega-MUSCLES

---

#### GJ 729 (Ross 154)

- **Star name:** `gj729`
- **URL:** https://simbad.cds.unistra.fr/simbad/sim-id?Ident=GJ+729
- **Spectral type:** M3.5V
- **Teff:** ~3200–3300 K
- **Age:** <1–2 Gyr
- **Luminosity:** ~0.004–0.005 L☉
- **Mass:** ~0.18 M☉
- **Radius:** ~0.20 R☉
- **Source:** Mega-MUSCLES

---

#### GJ 832

- **Star name:** `gj832`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/HIP%20106440
- **Spectral type:** M1-M3V
- **Teff:** ~3500-3500 K
- **Age:** ~4–12 Gyr
- **Luminosity:** ~0.03 L☉
- **Mass:** 0.45 M☉
- **Radius:** 0.45 R☉
- **Source:** MUSCLES

---

#### GJ 876

- **Star name:** `gj876`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20876
- **Spectral type:** M2-4V
- **Teff:** ~3200–3300 K
- **Age:** ~2–15 Gyr
- **Luminosity:** ~0.013 L☉
- **Mass:** ~0.37 M☉
- **Radius:** ~0.37 R☉
- **Source:** MUSCLES

---

#### HAT-P-12

- **Star name:** `hat-p-12`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/HAT-P-12
- **Spectral type:** K4V
- **Teff:** 4500-4800 K
- **Age:** 2-14 Gyr
- **Luminosity:** ~0.20 L☉
- **Mass:** 0.74 M☉
- **Radius:** 0.70 R☉
- **Source:** MUSCLES extension

---

#### HAT-P-26

- **Star name:** `hat-p-26`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/HAT-P-26
- **Spectral type:** K1V
- **Teff:** ~5050 K
- **Age:** 4-12 Gyr
- **Luminosity:** ~0.44 L☉
- **Mass:** 0.85 M☉
- **Radius:** 0.86 R☉
- **Source:** MUSCLES extension

---

#### HD 149026

- **Star name:** `hd-149026`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/HD%20149026%20b#planet_HD-149026
- **Spectral type:** G0V
- **Teff:** ~6100-6200 K
- **Age:** ~2-3 Gyr
- **Luminosity:** ~2.6 L☉
- **Mass:** ~1.14 M☉
- **Radius:** ~1.46 R☉
- **Source:** MUSCLES extension

---

#### HD 40307

- **Star name:** `hd40307`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/HD%2040307
- **Spectral type:** K2.5V
- **Teff:** ~4800-5000 K
- **Age:** ~2–5 Gyr
- **Luminosity:** ~0.22 L☉
- **Mass:** ~0.79 M☉
- **Radius:** ~0.71 R☉
- **Source:** MUSCLES

---

#### L 678-39 (GJ 357)

- **Star name:** `l-678-39`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/GJ%20357
- **Spectral type:** M2.5V
- **Teff:** ~3500 K
- **Age:** ?
- **Luminosity:** 0.0017 L☉
- **Mass:** ~0.34 M☉
- **Radius:** ~0.34 R☉
- **Source:** MUSCLES extension

---

#### L 980-5

- **Star name:** `l-980-5`
- **URL:** https://simbad.u-strasbg.fr/simbad/sim-id?Ident=l+980-5
- **Spectral type:** M4V
- **Teff:** ?
- **Age:** ?
- **Luminosity:** ?
- **Mass:** ?
- **Radius:** ?
- **Source:** Mega-MUSCLES

---

#### LHS 2686

- **Star name:** `lhs-2686`
- **URL:** https://simbad.cds.unistra.fr/simbad/sim-id?Ident=LHS+2686
- **Spectral type:** M5V
- **Teff:** ?
- **Age:** ?
- **Luminosity:** ?
- **Mass:** ?
- **Radius:** ?
- **Source:** Mega-MUSCLES

---

#### LP 791-18

- **Star name:** `lp-791-18`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/LP%20791-18
- **Spectral type:** M6V
- **Teff:** ~2960 K
- **Age:** > 0.5 Gyr
- **Luminosity:** ~0.002 L☉
- **Mass:** 0.139 M☉
- **Radius:** 0.182 R☉
- **Source:** MUSCLES extension

---

### TOI-193 (LTT 9779)

- **Star name:** `toi-193`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/LTT%209779
- **Spectral type:** G7V
- **Teff:** ~5400-5500 K
- **Age:** ~2 Gyr
- **Luminosity:** ~ 0.7 L☉
- **Mass:** ~1.0 M☉
- **Radius:** ~0.95 R☉
- **Source:** MUSCLES extension

---

### WASP-127

- **Star name:** `wasp-127`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/WASP-127
- **Spectral type:** G5 (subgiant-like)
- **Teff:** ~5600-5800 K
- **Age:** ~10–12 Gyr
- **Luminosity:** ~1.8 L☉
- **Mass:** ~0.95–1.10 M☉
- **Radius:** ~1.3 R☉
- **Source:** MUSCLES extension

---

### WASP-17

- **Star name:** `wasp-17`
- **URL:** https://exoplanetarchive.ipac.caltech.edu/overview/WASP-17
- **Spectral type:** F4–F6V
- **Teff:** ~6550 K
- **Age:** ~3 Gyr
- **Luminosity:** ~4 L☉
- **Mass:** 1.35 M☉
- **Radius:** 1.57 R☉
- **Source:** MUSCLES extension


### Using a PHOENIX synthetic spectrum

To use a Med-Res PHOENIX synthetic spectrum, set:

- `spectrum_source = "phoenix"`

PHOENIX spectra are stored under:
- `$FWL_DATA/stellar_spectra/PHOENIX/`

Each subdirectory corresponds to a metallicity / alpha combination, e.g.
`FeH-0.5_alpha+0.0/`.

#### Parameters

PHOENIX models are defined on a grid in:

- `Teff` (K) — effective temperature
- `log_g` (dex) — surface gravity
- `FeH` (dex) — metallicity
- `alpha` (dex) — alpha enhancement

In addition, PROTEUS needs the stellar radius:

- `radius` (R☉)

This is used to scale the model spectrum (surface flux) to the flux at 1 AU.

You can set these under `star.mors` in your config file.

**Defaults / fallbacks**
- If `FeH` and/or `alpha` are not set, they default to solar (0.0).
- If `Teff`, `log_g`, and/or `radius` are not set, they are estimated by the stellar evolution module (`mors`) from the stellar mass (if provided).


### Using a custom stellar spectrum

If you prefer to use your own stellar spectrum, you can input its filepath under the parameter `star_path`. Make sure you input its absolute path, even if it is in the $FWL_DATA directory. **NOTE**: this parameter will override all other stellar spectrum config parameters!

There are a few things to take into account when using a custom stellar spectrum:

- The file should be a two-column ASCII file, with the first column the **wavelength** in **nm**, and the second column the **flux** in **erg/s/cm^2/nm**. Headers should be indicated with `#`.
- The spectrum must be **scaled to 1 AU**.

## Surfaces
Single-scattering albedo data taken from Hammond et al., (2024): [Zenodo data](https://zenodo.org/records/13691960).
Available options can be found by running the command:
```console
ls $FWL_DATA/surface_albedos/Hammond24
```

## Exoplanet population data
These are obtained from the DACE PlanetS database ([Parc et al., 2024](https://arxiv.org/abs/2406.04311)).

## Mass-radius relations
These are obtained from [Zeng et al., 2019](https://iopscience.iop.org/article/10.3847/0004-637X/819/2/127/meta).
