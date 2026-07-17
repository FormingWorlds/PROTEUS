"""Reference data and configuration for the petitRADTRANS observation module.

petitRADTRANS is an optional component. Nothing here is imported unless a config selects
``observe.module = "petitRADTRANS"``, and a machine without the package or its opacity
tables is a supported configuration.

Two jobs live here:

- fetching the opacity tables into ``$FWL_DATA/prt/input_data`` from the upstream library
  the petitRADTRANS team publishes;
- writing the petitRADTRANS configuration file, which names one opacity file per species
  so the library never stops to ask which file to use.

The library serves files over plain HTTP, so neither job needs a browser.
"""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request
from pathlib import Path

from proteus.utils.constants import prt_gases
from proteus.utils.data import GetFWLData

log = logging.getLogger('fwl.' + __name__)

# The upstream petitRADTRANS input-data library, a Keeper (Seafile) share.
PRT_LIBRARY_TOKEN = 'ccf25082fda448c8a0d0'
PRT_LIBRARY_HOST = 'https://keeper.mpdl.mpg.de'
PRT_LIBRARY_FILE_URL = f'{PRT_LIBRARY_HOST}/d/{PRT_LIBRARY_TOKEN}/files/?p='

# Opening bytes of an HDF5 file. The library answers a request for a path it does not hold
# with a web page and a success status rather than an error, so what came back has to be
# looked at: without this, a table renamed upstream would be saved as an opacity file and
# counted as installed, and petitRADTRANS would fail reading it far from the cause.
HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'

# Line species PROTEUS models. Derived from the gas list the observe module iterates over,
# so a gas added to prt_gases is covered here without a second edit.
PRT_LINE_SPECIES: tuple[str, ...] = tuple(prt_gases)

# The opacity file used for each species, keyed by its path within the library.
#
# These are choices, not defaults. Several species publish more than one table: water
# alone offers a HITEMP and a POKAZATEL line list over different wavelength ranges, and
# they do not produce the same spectrum. The entries below reproduce the set PROTEUS has
# been running with, made explicit here rather than left implicit in an install script.
#
# Naming a file per species is also what keeps petitRADTRANS from asking. Faced with
# several candidates and no instruction, it prompts for a choice, which on an unattended
# machine waits rather than fails.
#
# Swapping any entry changes the spectra PROTEUS produces and is a spectroscopic decision
# to take deliberately. Do not replace this table with a rule such as largest-file or
# newest-wins: those would drift the science quietly, which is the failure this constant
# exists to prevent.
PRT_DEFAULT_FILES: dict[str, str] = {
    # Line opacities
    'opacities/lines/correlated_k/C2H2/12C2-1H2': (
        '12C2-1H2__aCeTY.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/C2H4/12C2-1H4': (
        '12C2-1H4__MaYTY.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/CH4/12C-1H4': (
        '12C-1H4__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/CO/12C-16O': (
        '12C-16O__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/CO2/12C-16O2': (
        '12C-16O2__UCL-4000.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/H2/1H2': (
        '1H2__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/H2O/1H2-16O': (
        '1H2-16O__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/H2S/1H2-32S': (
        '1H2-32S__AYT2.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/HCN/1H-12C-14N': (
        '1H-12C-14N__Harris.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/NH3/14N-1H3': (
        '14N-1H3__CoYuTe.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/O/16O': (
        '16O__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/O2/16O2': (
        '16O2__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/O3/16O3': (
        '16O3__HITRAN.R1000_0.1-250mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/OH/16O-1H': (
        '16O-1H__MoLLIST.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/SH/32S-1H': (
        '32S-1H__GYT.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/SO2/32S-16O2': (
        '32S-16O2__ExoAmes.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/Si/28Si': (
        '28Si__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/SiO/28Si-16O': (
        '28Si-16O__SiOUVenIR.R1000_0.1-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/SiO2/28Si-16O2': (
        '28Si-16O2__OYT3.R1000_0.3-50mu.ktable.petitRADTRANS.h5'
    ),
    'opacities/lines/correlated_k/Si_+/28Si_+': (
        '28Si_p__Kurucz.R1000_0.1-250mu.ktable.petitRADTRANS.h5'
    ),
    # Collision-induced absorption
    'opacities/continuum/collision_induced_absorptions/CO2--CO2/C-O2--C-O2-NatAbund': (
        'C-O2--C-O2-NatAbund.DeltaWavelength1e-6_3-100mu.ciatable.petitRADTRANS.h5'
    ),
    'opacities/continuum/collision_induced_absorptions/H2--H2/H2--H2-NatAbund': (
        'H2--H2-NatAbund__BoRi.R831_0.6-250mu.ciatable.petitRADTRANS.h5'
    ),
    'opacities/continuum/collision_induced_absorptions/H2--He/H2--He-NatAbund': (
        'H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu.ciatable.petitRADTRANS.h5'
    ),
    'opacities/continuum/collision_induced_absorptions/H2O--H2O/H2-O--H2-O-NatAbund': (
        'H2-O--H2-O-NatAbund.DeltaWavenumber10_0.5-77mu.ciatable.petitRADTRANS.h5'
    ),
    'opacities/continuum/collision_induced_absorptions/H2O--N2/H2-O--N2-NatAbund': (
        'H2-O--N2-NatAbund.DeltaWavenumber10_0.5-77mu.ciatable.petitRADTRANS.h5'
    ),
    'opacities/continuum/collision_induced_absorptions/N2--H2/N2--H2-NatAbund': (
        'N2--H2-NatAbund.DeltaWavenumber1_5.3-909mu.ciatable.petitRADTRANS.h5'
    ),
    'opacities/continuum/collision_induced_absorptions/N2--He/N2--He-NatAbund': (
        'N2--He-NatAbund.DeltaWavenumber1_10-909mu.ciatable.petitRADTRANS.h5'
    ),
    'opacities/continuum/collision_induced_absorptions/N2--N2/N2--N2-NatAbund': (
        'N2--N2-NatAbund.DeltaWavelength1e-6_2-100mu.ciatable.petitRADTRANS.h5'
    ),
    'opacities/continuum/collision_induced_absorptions/N2--O2/N2--O2-NatAbund': (
        'N2--O2-NatAbund.DeltaWavelength1e-6_0.72-5.4mu.ciatable.petitRADTRANS.h5'
    ),
    'opacities/continuum/collision_induced_absorptions/O2--O2/O2--O2-NatAbund': (
        'O2--O2-NatAbund.DeltaWavelength1e-6_0.34-8.7mu.ciatable.petitRADTRANS.h5'
    ),
}

# Bytes per read while streaming a table to disk.
_CHUNK = 1 << 20


def _config_path() -> Path:
    """Return the path petitRADTRANS reads its configuration from."""
    return Path.home() / '.petitradtrans' / 'petitradtrans_config_file.ini'


def prt_input_data_dir() -> Path:
    """Return the directory holding the petitRADTRANS opacity tables."""
    return GetFWLData() / 'prt' / 'input_data'


def _expected_files() -> dict[Path, str]:
    """Map each table's destination on disk to its path within the library."""
    root = prt_input_data_dir()
    return {root / sub / name: f'/{sub}/{name}' for sub, name in PRT_DEFAULT_FILES.items()}


def missing_tables() -> list[Path]:
    """Return the opacity tables named in PRT_DEFAULT_FILES that are not on disk."""
    return [dest for dest in _expected_files() if not dest.is_file()]


def opacities_present() -> bool:
    """True when every table PROTEUS asks for is on disk.

    Completeness is judged against the named set rather than against whatever happens to
    be in the directory. A run interrupted part-way leaves some tables behind, and calling
    that present would strand the tree: nothing would fetch the rest, and petitRADTRANS
    would go looking for them over the network mid-run.
    """
    return not missing_tables()


def uncovered_species() -> list[str]:
    """Return the gases PROTEUS may model that no opacity table is named for.

    These are not an error. The observe module drops a gas with no table from the
    radiative transfer and carries on, so the spectrum is computed without those
    opacities rather than failing. The list is reported so the omission is visible.
    """
    named = {
        sub.split('/')[3] for sub in PRT_DEFAULT_FILES if sub.startswith('opacities/lines/')
    }
    return [species for species in PRT_LINE_SPECIES if species not in named]


def _download(library_path: str, destination: Path) -> None:
    """Stream one table out of the library, leaving nothing behind unless it is a table.

    The file lands under a temporary name and is moved into place only once it has
    arrived in full and been recognised as HDF5, so neither a truncated transfer nor a
    web page returned in place of a table can leave something behind that later looks
    installed.

    Raises
    ------
    OSError
        If the transfer fails, or if what came back is not an opacity table.
    """
    url = PRT_LIBRARY_FILE_URL + urllib.parse.quote(library_path, safe='') + '&dl=1'
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            content_type = response.headers.get('Content-Type', '')
            if 'html' in content_type.lower():
                raise OSError(
                    f'The library answered with a web page rather than a table for '
                    f'{library_path}. The file is most likely no longer at that path.'
                )
            with open(partial, 'wb') as handle:
                while chunk := response.read(_CHUNK):
                    handle.write(chunk)

        with open(partial, 'rb') as handle:
            if handle.read(len(HDF5_MAGIC)) != HDF5_MAGIC:
                raise OSError(f'What arrived for {library_path} is not an HDF5 table.')

        partial.rename(destination)
    finally:
        partial.unlink(missing_ok=True)


def download_prt_opacities(clean: bool = False) -> bool:
    """Fetch the petitRADTRANS opacity tables PROTEUS needs.

    Only the tables named in PRT_DEFAULT_FILES are fetched, and only those not already on
    disk, so an interrupted run resumes where it stopped rather than starting over. The
    configuration is written afterwards, because tables petitRADTRANS cannot locate are of
    no use.

    Parameters
    ----------
    clean : bool
        Fetch every table again, even those already present.

    Returns
    -------
    bool
        True when every table is in place afterwards.
    """
    expected = _expected_files()
    wanted = list(expected) if clean else missing_tables()

    if not wanted:
        log.debug('petitRADTRANS opacity tables already present')
        write_prt_config()
        return True

    log.info('Downloading %d petitRADTRANS opacity tables (several GB)', len(wanted))

    for dest in wanted:
        library_path = expected[dest]
        log.debug('Fetching %s', library_path)
        try:
            _download(library_path, dest)
        except Exception as exc:
            log.warning('Failed to fetch %s: %s', library_path, exc)
            return False

    absent = missing_tables()
    if absent:
        log.warning('petitRADTRANS opacity tables still missing: %s', [p.name for p in absent])
        return False

    write_prt_config()
    return True


def write_prt_config() -> Path:
    """Write the petitRADTRANS configuration file and return its path.

    Records where the tables live and names the file to use for each species.

    Raises
    ------
    FileNotFoundError
        If any table PROTEUS asks for is absent, since a configuration naming files that
        are not there would send petitRADTRANS to fetch them mid-run.
    """
    root = prt_input_data_dir()
    absent = missing_tables()
    if absent:
        raise FileNotFoundError(
            f'{len(absent)} petitRADTRANS opacity tables are missing under {root}, '
            f'starting with {absent[0].name}. Fetch them with '
            'proteus.utils.prt_data.download_prt_opacities() first.'
        )

    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ['[Default files]', '']
    lines += [f'{sub} = {name}' for sub, name in sorted(PRT_DEFAULT_FILES.items())]
    lines += [
        '',
        '[Paths]',
        f'prt_input_data_path = {root}',
        '',
        '[URLs]',
        f'prt_input_data_url = {PRT_LIBRARY_FILE_URL}',
        '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')

    log.debug(
        'Wrote petitRADTRANS config to %s with %d named files', path, len(PRT_DEFAULT_FILES)
    )
    return path
