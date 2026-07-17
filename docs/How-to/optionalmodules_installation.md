# Installation: optional modules

!!! info "Install PROTEUS first"
    Please follow the [quick installation guide](installation.md) or [manual installation guide](manual_installation.md) before installing optional modules.

## SPIDER and PETSc {#optional-setup-petsc}

SPIDER is a C-based interior thermal evolution solver. It requires PETSc
(a numerical computing library) and a C compiler. Most configurations use
Aragog instead, which is written in Python/JAX and needs no additional
compiled dependencies.

!!! warning
    PETSc requires Python <= 3.12. Make sure your conda environment uses
    Python 3.12.

```console
bash tools/get_petsc.sh
bash tools/get_spider.sh
```

=== "Linux"

    The PETSc script downloads a pre-compiled version from OSF and configures
    `PETSC_DIR` and `PETSC_ARCH` automatically.

    !!! note "Fedora / RHEL"
        If you encounter errors moving libraries, see
        [Troubleshooting: PETSc on Fedora/RHEL](troubleshooting.md#cannot-compile-petsc-error-moving-libraries-fedorarhel).

=== "macOS"

    The script detects Apple Silicon vs Intel and uses Homebrew's MPI. If you
    encounter issues, see
    [Troubleshooting: PETSc on Apple Silicon](troubleshooting.md#petsc-compilation-fails-on-apple-silicon).

## Multi-phase tidal heating (LovePy)

LovePy computes tidal dissipation for multi-phase planetary interiors. It is
written in Julia.

```console
bash tools/get_lovepy.sh
```

## Synthetic observations (petitRADTRANS)

[petitRADTRANS](https://petitradtrans.readthedocs.io/en/latest/) generates
synthetic transmission and secondary eclipse spectra. It is selected with
`observe.module = "petitRADTRANS"`.

Installing it takes three steps.

**1. Install the package.** It builds Fortran, C and C++ extensions from
source, so the host needs `gfortran` and a C/C++ toolchain before you start.

```console
pip install -e ".[observe]"
```

**2. Fetch the opacity tables.** About 5.6 GB, so allow time and disk. They
land in `$FWL_DATA/prt/input_data` and are fetched once; a run interrupted
part-way picks up where it stopped.

```console
python -c "from proteus.utils.prt_data import download_prt_opacities; download_prt_opacities()"
```

**3. Write the configuration.** This tells petitRADTRANS where the tables are
and which file to use for each species. Step 2 does it for you; run this if
you move `FWL_DATA` or want to rewrite the file.

```console
bash tools/get_prt.sh
```

### Citing the opacity data

The tables are line lists produced by other groups and processed into
correlated-k form by the petitRADTRANS team. If you publish spectra, cite the
line list for each species you used, alongside petitRADTRANS itself
([Mollière et al. 2019](https://doi.org/10.1051/0004-6361/201935470)).

| Species | Source | Reference |
|---|---|---|
| H<sub>2</sub>O, CO | HITEMP | [Rothman et al. 2010](https://doi.org/10.1016/j.jqsrt.2010.05.001) |
| CH<sub>4</sub> | HITEMP | [Hargreaves et al. 2020](https://doi.org/10.3847/1538-4365/ab7a1a) |
| H<sub>2</sub>, O<sub>3</sub> | HITRAN | [Rothman et al. 2013](https://doi.org/10.1016/j.jqsrt.2013.07.002) |
| O<sub>2</sub> | HITRAN | [Gordon et al. 2022](https://doi.org/10.1016/j.jqsrt.2021.107949) |
| C<sub>2</sub>H<sub>2</sub> | ExoMol aCeTY | [Chubb et al. 2020](https://doi.org/10.1093/mnras/staa229) |
| C<sub>2</sub>H<sub>4</sub> | ExoMol MaYTY | [Mant et al. 2018](https://doi.org/10.1093/mnras/sty1239) |
| H<sub>2</sub>S | ExoMol AYT2 | [Azzam et al. 2016](https://doi.org/10.1093/mnras/stw1133) |
| HCN | ExoMol Harris | [Barber et al. 2013](https://doi.org/10.1093/mnras/stt2011) |
| NH<sub>3</sub> | ExoMol CoYuTe | [Coles et al. 2019](https://doi.org/10.1093/mnras/stz2778) |
| SO<sub>2</sub> | ExoMol ExoAmes | [Underwood et al. 2016](https://doi.org/10.1093/mnras/stw849) |
| SH | ExoMol GYT | [Yurchenko et al. 2018](https://doi.org/10.1093/mnras/sty939) |
| SiO | ExoMol SiOUVenIR | [Yurchenko et al. 2021](https://doi.org/10.1093/mnras/stab3267) |
| OH | MoLLIST | [Yousefi et al. 2018](https://doi.org/10.1016/j.jqsrt.2018.06.016) |

Each reference above is the one recorded inside the table itself.

Three groups of tables need care, because what they record is not enough to
cite from.

**Tables giving a source but no formal citation.** SiO<sub>2</sub> records the
ExoMol OYT3 line list, and O, Si and Si<sup>+</sup> record the
[Kurucz atomic line lists](http://kurucz.harvard.edu/), each as a bare URL.
Cite those sources directly.

**Tables recording nothing usable.** CO<sub>2</sub> carries a placeholder where
its reference should be, and the collision-induced absorption tables for
CO<sub>2</sub>--CO<sub>2</sub>, H<sub>2</sub>O--H<sub>2</sub>O,
H<sub>2</sub>O--N<sub>2</sub>, N<sub>2</sub>--H<sub>2</sub>,
N<sub>2</sub>--He, N<sub>2</sub>--N<sub>2</sub>, N<sub>2</sub>--O<sub>2</sub>
and O<sub>2</sub>--O<sub>2</sub> record no source at all. The CO<sub>2</sub>
filename points to the ExoMol UCL-4000 line list, but that is read off the name
rather than confirmed by the table, so check it against petitRADTRANS' own
[opacity documentation](https://petitradtrans.readthedocs.io/en/latest/) before
citing it.

**Tables recording petitRADTRANS in place of the data's own source.** The
H<sub>2</sub>--H<sub>2</sub> and H<sub>2</sub>--He collision-induced absorption
tables record the petitRADTRANS release paper as their reference. That is the
code, not the origin of the cross-sections, which come from a third-party
compilation. Cite the primary source from petitRADTRANS' opacity documentation
rather than the reference the file carries.

ExoMol data are released under
[CC BY-SA 4.0](https://exomol.com/data/licence/). HITRAN asks that you cite the
database edition you used; see its
[citation policy](https://hitran.org/citepolicy/).

### Which species are covered

`proteus.utils.prt_data.PRT_DEFAULT_FILES` names the file used for each
species. Several species publish more than one line list, and they do not give
the same spectrum, so the choice is written out rather than left to a rule.
Changing an entry changes your spectra.

A gas with no table is dropped from the radiative transfer and its opacity is
absent from the result. `proteus.utils.prt_data.uncovered_species()` reports
any such gas.

## Alternative outgassing (atmodeller)

[atmodeller](https://github.com/djbower/atmodeller) is an optional
alternative to CALLIOPE, selected with `outgas.module = "atmodeller"`. It is
not installed with PROTEUS by default; a standard run uses CALLIOPE.

```console
pip install "fwl-proteus[atmodeller]"
```

!!! warning "License"
    atmodeller is distributed under the GPL-3.0 license; review its terms
    before installing.

## Atmospheric chemistry (VULCAN)

VULCAN is an optional atmospheric-photochemistry backend, selected with
`atmos_chem.module = "vulcan"`. It is not required for a standard PROTEUS
run. Install it from PyPI:

```console
pip install "fwl-proteus[vulcan]"
```

For local development, install as an editable checkout instead:

```console
bash tools/get_vulcan.sh
```

!!! warning "License"
    VULCAN is distributed under the GPL-3.0 license; review its terms
    before installing.
