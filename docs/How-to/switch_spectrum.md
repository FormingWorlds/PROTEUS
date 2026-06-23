# Switch and choose a stellar spectrum

PROTEUS supports four stellar spectrum sources, selected via
`star.mors.spectrum_source` in your config file. This guide walks through
each option and when to use it.

!!! info "Applies to `module = "mors"` only"
    Spectrum selection is only available when `star.module = "mors"`.
    The dummy stellar module uses a fixed blackbody and ignores these settings.

## Which source should I use?

| Source | Best for |
|---|---|
| `"solar"` | Simulating the Sun or a planet in the Solar System |
| `"muscles"` | A real M, K, or G star with an observed UV/X-ray spectrum |
| `"phoenix"` | Any star without an observed spectrum; flexible grid coverage |
| `star_path` | Your own spectrum file from any source |

When in doubt, use `"phoenix"`. It covers the widest range of stellar
parameters and is downloaded automatically.

---

## Option 1: Solar spectrum

Use this when your host star is the Sun (modern or at a past/future age).

```toml
[star.mors]
    spectrum_source = "solar"
    star_name       = "sun"       # modern NREL solar spectrum
```

To use a **young or future Sun**, choose one of the
[VPL spectra](https://live-vpl-test.pantheonsite.io/models/evolution-of-solar-flux/)
by Claire et al. (2012):

```toml
[star.mors]
    spectrum_source = "solar"
    star_name       = "Sun3.8Ga"  # Sun as it was 3.8 Gyr ago (age ~ 800 Myr)
```

Available `star_name` values for the solar source (case-insensitive):

| `star_name` | Description |
|---|---|
| `sun` | Modern NREL spectrum (default) |
| `SunModern` | Alternative modern solar spectrum |
| `Sun0.6Ga` | 0.6 Gyr ago, age ~ 4.0 Gyr |
| `Sun1.8Ga` | 1.8 Gyr ago, age ~ 2.8 Gyr |
| `Sun2.4Ga` | 2.4 Gyr ago, age ~ 2.2 Gyr |
| `Sun2.7Ga` | 2.7 Gyr ago, age ~ 1.9 Gyr |
| `Sun3.8Ga` | 3.8 Gyr ago, age ~ 800 Myr |
| `Sun4.4Ga` | 4.4 Gyr ago, age ~ 200 Myr |
| `Sun5.6Gyr` | Future Sun, age 5.6 Gyr |

### Downloading solar spectra

Solar spectra are downloaded automatically on first run. You can also
pre-download manually with the CLI:

```bash
proteus get solar
```

---

## Option 2: MUSCLES / Mega-MUSCLES observed spectrum

Use this when your host star is a real star covered by the
[MUSCLES](https://archive.stsci.edu/prepds/muscles/) or
[Mega-MUSCLES](https://archive.stsci.edu/prepds/mega-muscles/) surveys.
These spectra include observed UV and X-ray flux, which matters most for
photochemistry and escape calculations.

```toml
[star.mors]
    spectrum_source = "muscles"
    star_name       = "trappist-1"
```

The full list of available stars is in the
[Reference data](../Reference/data.md#muscles-mega-muscles-spectra)
page. `star_name` is case-insensitive; the names listed in the reference
catalog are the canonical forms (e.g. `gj551` for Proxima Centauri,
`v-eps-eri` for Epsilon Eridani).

### Downloading MUSCLES spectra

Spectra are downloaded automatically on first run. You can also pre-download
manually with the CLI:

```bash
# List all available star names
proteus get muscles --list

# Single star
proteus get muscles --star trappist-1

# All available stars
proteus get muscles --all
```

---

## Option 3: PHOENIX synthetic spectrum

Use this when no observed spectrum exists for your target star. PHOENIX
provides a grid of synthetic spectra over a wide range of stellar parameters.

```toml
[star.mors]
    spectrum_source = "phoenix"
```

By default, `Teff`, `log_g`, and `radius` are derived from the stellar
mass via the MORS stellar evolution tracks, and metallicity defaults to
solar. To override any of these:

```toml
[star.mors]
    spectrum_source  = "phoenix"
    phoenix_Teff     = 3500       # effective temperature [K]
    phoenix_log_g    = 5.0        # surface gravity [log10 cgs]
    phoenix_radius   = 0.12       # stellar radius [R_sun]
    phoenix_FeH      = 0.0        # metallicity [Fe/H]; 0.0 = solar
    phoenix_alpha    = 0.0        # alpha enhancement; 0.0 = solar
```

!!! info "Defaults"
    Any `phoenix_*` parameter set to `"none"` is derived from the MORS
    stellar evolution tracks using the configured stellar `mass`.
    `phoenix_FeH` and `phoenix_alpha` default to `0.0` (solar) when unset.

!!! warning "Stellar mass limits"
    The MORS tracks used to derive PHOENIX parameters cover a limited mass
    range. If `star.mass` falls outside this range it is silently clipped
    with a warning before the spectrum is built:

    | Track (`star.mors.tracks`) | Valid range |
    |---|---|
    | `spada` | 0.10 – 1.25 M☉ |
    | `baraffe` | 0.01 – 1.40 M☉ |

### Downloading PHOENIX spectra

PHOENIX spectra are downloaded automatically on first run. To pre-download
for a specific parameter combination, `--feh` and `--alpha` are required;
`--teff` is optional but helps select the correct alpha availability:

```bash
proteus get phoenix --feh 0.0 --alpha 0.0 --teff 3500
```

---

## Option 4: Custom spectrum file

Use this when you have your own spectrum file that PROTEUS should use
directly, regardless of what `spectrum_source` is set to.

```toml
[star.mors]
    star_path = "/absolute/path/to/my_spectrum.txt"
    # or, if the file is inside $FWL_DATA:
    star_path = "$FWL_DATA/stellar_spectra/my_spectrum.txt"
```

!!! warning "`star_path` overrides everything"
    Setting `star_path` bypasses `spectrum_source`, `star_name`, and all
    `phoenix_*` parameters entirely.

Your spectrum file must be:

- A **two-column ASCII** file (comments with `#`)
- Column 1: wavelength in **nm**
- Column 2: flux in **erg s$^{-1}$ cm$^{-2}$ nm$^{-2}$**, **scaled to 1 AU**