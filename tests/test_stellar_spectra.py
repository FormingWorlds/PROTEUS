# tests/test_spectrum_pipeline.py
#
# Tests for the PROTEUS stellar spectrum pipeline:
# - phoenix_helper grid/filename/param stuff
# - PHOENIX raw -> scaled to 1 AU path creation (fake downloads)
# - init_star() spectrum selection (solar / MUSCLES / star_path / PHOENIX)
#
# Can run offline and without the real mors module
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# tests/test_stellar_spectra.py



# Helpers

def _write_spectrum_file(path: Path, wl=(100.0, 200.0), fl=(1.0, 2.0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([np.array(wl), np.array(fl)]))


def _make_handler(
    *,
    tmp_path: Path,
    offline: bool,
    Teff: float = 5800.0,
    logg: float = 4.5,
    radius: float = 1.0,
    FeH: float = 0.0,
    alpha: float = 0.0,
    tracks: str = "spada",
    age_now: float = 4.6,  # Gyr
    mass: float = 1.0,
):
    mors = SimpleNamespace(
        age_now=age_now,
        phoenix_FeH=FeH,
        phoenix_alpha=alpha,
        tracks=tracks,
        phoenix_Teff=Teff,
        phoenix_radius=radius,
        phoenix_log_g=logg,
    )
    star = SimpleNamespace(mass=mass, mors=mors)
    params = SimpleNamespace(offline=offline)
    config = SimpleNamespace(star=star, params=params)

    outdir = tmp_path / "out"
    (outdir / "data").mkdir(parents=True, exist_ok=True)

    handler = SimpleNamespace(
        config=config,
        directories={
            "fwl": str(tmp_path),
            "output": str(outdir),              # <-- add this
            "output/data": str(outdir / "data") # <-- and this
        },
    )
    return handler


def _raw_phoenix_path(tmp_path: Path, raw_name: str, FeH_str: str, alpha_str: str) -> Path:
    return (
        tmp_path
        / "stellar_spectra"
        / "PHOENIX"
        / f"FeH{FeH_str}_alpha{alpha_str}"
        / raw_name
    )


def _install_fake_mors(monkeypatch):
    """
    Minimal fake 'mors' module so proteus.star.wrapper.init_star() can run
    through the Baraffe branch without the real mors install.
    """

    class FakeBaraffeTrack:
        def __init__(self, Mstar):
            self.Mstar = float(Mstar)

        def BaraffeLuminosity(self, age_yr):
            return 1.0

        # not required by init_star, but harmless if accessed
        def BaraffeSolarConstant(self, age_yr, sep_au):
            return 1361.0

        def BaraffeStellarRadius(self, age_yr):
            return 1.0

        def BaraffeStellarTeff(self, age_yr):
            return 5800.0

    def ModernSpectrumLoad(star_modern_path, star_backup_path):
        data = np.loadtxt(star_modern_path)
        return data[:, 0], data[:, 1]

    fake = SimpleNamespace(
        BaraffeTrack=FakeBaraffeTrack,
        ModernSpectrumLoad=ModernSpectrumLoad,
    )
    monkeypatch.setitem(sys.modules, "mors", fake)
    return fake


def _make_handler_for_init_star(
    tmp_path: Path,
    *,
    spectrum_source,
    star_name="gj 876",
    tracks="baraffe",
    star_path=None,
    offline=True,
):
    mors = SimpleNamespace(
        star_path=star_path,
        star_name=star_name,
        spectrum_source=spectrum_source,
        tracks=tracks,
        age_now=4.6,      # Gyr
        rot_pcntle=50.0,
        rot_period=None,
        phoenix_FeH=0.0,
        phoenix_alpha=0.0,
        phoenix_Teff=None,
        phoenix_radius=None,
        phoenix_log_g=None,
    )
    star = SimpleNamespace(module="mors", mass=1.0, mors=mors, bol_scale=1.0)
    params = SimpleNamespace(offline=offline)
    config = SimpleNamespace(star=star, params=params)

    outdir = tmp_path / "out"
    (outdir / "data").mkdir(parents=True, exist_ok=True)

    handler = SimpleNamespace(
        config=config,
        directories={
            "fwl": str(tmp_path),
            "output": str(outdir),
            "output/data": str(outdir / "data")
        },
    )
    return handler

# phoenix_helper.py tests

def test_phoenix_param_zero_formatting():
    from proteus.utils.phoenix_helper import phoenix_param

    assert phoenix_param(0.0, kind="FeH") == "-0.0"
    assert phoenix_param(0.0, kind="alpha") == "+0.0"


def test_phoenix_to_grid_snaps_to_nearest():
    from proteus.utils.phoenix_helper import phoenix_to_grid

    grid = phoenix_to_grid(FeH=-0.12, alpha=0.33, Teff=5873, logg=4.62)
    assert grid["FeH"] == 0.0
    assert grid["alpha"] == pytest.approx(0.4)
    assert grid["Teff"] == 5900.0
    assert grid["logg"] == 4.5


def test_phoenix_to_grid_disallows_alpha_when_feh_positive(caplog):
    """
    Use FeH that snaps to +0.5 so alpha is not allowed.
    (FeH=+0.2 would snap to 0.0, and then alpha *is* allowed.)
    """
    from proteus.utils.phoenix_helper import phoenix_to_grid

    caplog.set_level("WARNING")
    grid = phoenix_to_grid(FeH=+0.4, alpha=0.6, Teff=5800, logg=4.5)

    assert grid["FeH"] == 0.5
    assert grid["alpha"] == 0.0
    assert any("using [alpha/M]=0.0" in rec.message for rec in caplog.records)


def test_phoenix_filename_format():
    from proteus.utils.phoenix_helper import phoenix_filename

    name = phoenix_filename(Teff=2300, logg=1.0, FeH=0.0, alpha=0.0)
    assert name == "LTE_T02300_logg1.00_FeH-0.0_alpha+0.0_phoenixMedRes_R05000.txt"


# star/phoenix.py tests

def test_get_phoenix_modern_spectrum_scales_to_1au(tmp_path, monkeypatch):
    import proteus.star.phoenix as phoenix_mod
    from proteus.star.phoenix import get_phoenix_modern_spectrum
    from proteus.utils.constants import AU, R_sun
    from proteus.utils.phoenix_helper import phoenix_filename, phoenix_param

    # MUST patch the imported symbol inside proteus.star.phoenix
    monkeypatch.setattr(phoenix_mod, "GetFWLData", lambda: tmp_path)

    handler = _make_handler(tmp_path=tmp_path, offline=True, Teff=5800.0, logg=4.5, radius=1.0)

    raw_name = phoenix_filename(5800.0, 4.5, 0.0, 0.0)
    feh_str = phoenix_param(0.0, kind="FeH")
    alpha_str = phoenix_param(0.0, kind="alpha")
    raw_path = _raw_phoenix_path(tmp_path, raw_name, feh_str, alpha_str)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    wl = np.array([100.0, 200.0])
    fl_surface = np.array([1.0e3, 2.0e3])
    np.savetxt(raw_path, np.column_stack([wl, fl_surface]))

    au_path = get_phoenix_modern_spectrum(handler, stellar_track=None)
    out = np.loadtxt(au_path)

    scale = ((1.0 * R_sun) / AU) ** 2
    assert np.allclose(out[:, 0], wl)
    assert np.allclose(out[:, 1], fl_surface * scale)


def test_get_phoenix_modern_spectrum_offline_missing_raw_raises(tmp_path, monkeypatch):
    import proteus.star.phoenix as phoenix_mod
    from proteus.star.phoenix import get_phoenix_modern_spectrum

    monkeypatch.setattr(phoenix_mod, "GetFWLData", lambda: tmp_path)
    handler = _make_handler(tmp_path=tmp_path, offline=True, Teff=5800.0, logg=4.5, radius=1.0)

    with pytest.raises(FileNotFoundError):
        get_phoenix_modern_spectrum(handler, stellar_track=None)


def test_get_phoenix_modern_spectrum_downloads_when_online(tmp_path, monkeypatch):
    import proteus.star.phoenix as phoenix_mod
    from proteus.star.phoenix import get_phoenix_modern_spectrum
    from proteus.utils.phoenix_helper import phoenix_filename, phoenix_param

    monkeypatch.setattr(phoenix_mod, "GetFWLData", lambda: tmp_path)
    handler = _make_handler(tmp_path=tmp_path, offline=False, Teff=5800.0, logg=4.5, radius=1.0)

    raw_name = phoenix_filename(5800.0, 4.5, 0.0, 0.0)
    feh_str = phoenix_param(0.0, kind="FeH")
    alpha_str = phoenix_param(0.0, kind="alpha")
    raw_path = _raw_phoenix_path(tmp_path, raw_name, feh_str, alpha_str)

    def fake_download_phoenix(*, alpha, FeH):
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(raw_path, np.column_stack([np.array([100.0, 200.0]), np.array([10.0, 20.0])]))
        return True

    monkeypatch.setattr(phoenix_mod, "download_phoenix", fake_download_phoenix)

    au_path = get_phoenix_modern_spectrum(handler, stellar_track=None)
    assert au_path.exists()
    assert raw_path.exists()


# init_star() spectrum selection tests

def test_init_star_source_none_prefers_muscles_when_available(tmp_path, monkeypatch):
    from proteus.star.wrapper import init_star

    _install_fake_mors(monkeypatch)
    handler = _make_handler_for_init_star(tmp_path, spectrum_source=None)

    star_file = "gj876.txt"
    solar = tmp_path / "stellar_spectra" / "solar" / star_file
    muscles = tmp_path / "stellar_spectra" / "MUSCLES" / star_file

    _write_spectrum_file(solar, fl=(10.0, 20.0))
    _write_spectrum_file(muscles, fl=(30.0, 40.0))

    init_star(handler)

    backup = tmp_path / "out" / "data" / "-1.sflux"
    arr = np.loadtxt(backup)
    assert np.allclose(arr[:, 1], np.array([30.0, 40.0]))


def test_init_star_source_none_uses_muscles_when_solar_missing(tmp_path, monkeypatch):
    from proteus.star.wrapper import init_star

    _install_fake_mors(monkeypatch)
    handler = _make_handler_for_init_star(tmp_path, spectrum_source=None)

    starname_proper = "gj876.txt"
    muscles = tmp_path / "stellar_spectra" / "MUSCLES" / starname_proper
    _write_spectrum_file(muscles, fl=(30.0, 40.0))

    init_star(handler)

    backup = tmp_path / "out" / "data" / "-1.sflux"
    arr = np.loadtxt(backup)
    assert np.allclose(arr[:, 1], np.array([30.0, 40.0]))


def test_init_star_source_solar_falls_back_to_muscles_with_warning(tmp_path, monkeypatch, caplog):
    from proteus.star.wrapper import init_star

    caplog.set_level("WARNING")
    _install_fake_mors(monkeypatch)
    handler = _make_handler_for_init_star(tmp_path, spectrum_source="solar")

    starname_proper = "gj876.txt"
    muscles = tmp_path / "stellar_spectra" / "MUSCLES" / starname_proper
    _write_spectrum_file(muscles, fl=(30.0, 40.0))

    init_star(handler)

    assert any("Requested solar spectrum" in r.message for r in caplog.records)
    backup = tmp_path / "out" / "data" / "-1.sflux"
    arr = np.loadtxt(backup)
    assert np.allclose(arr[:, 1], np.array([30.0, 40.0]))


def test_init_star_source_muscles_falls_back_to_solar_with_warning(tmp_path, monkeypatch, caplog):
    from proteus.star.wrapper import init_star

    caplog.set_level("WARNING")
    _install_fake_mors(monkeypatch)
    handler = _make_handler_for_init_star(tmp_path, spectrum_source="muscles")

    starname_proper = "gj876.txt"
    solar = tmp_path / "stellar_spectra" / "solar" / starname_proper
    _write_spectrum_file(solar, fl=(10.0, 20.0))

    init_star(handler)

    assert any("Requested MUSCLES spectrum" in r.message for r in caplog.records)
    backup = tmp_path / "out" / "data" / "-1.sflux"
    arr = np.loadtxt(backup)
    assert np.allclose(arr[:, 1], np.array([10.0, 20.0]))


def test_init_star_source_none_missing_both_raises(tmp_path, monkeypatch):
    from proteus.star.wrapper import init_star

    _install_fake_mors(monkeypatch)
    handler = _make_handler_for_init_star(tmp_path, spectrum_source=None)

    with pytest.raises(FileNotFoundError):
        init_star(handler)


def test_init_star_star_path_override_is_used(tmp_path, monkeypatch):
    from proteus.star.wrapper import init_star

    _install_fake_mors(monkeypatch)

    custom = tmp_path / "custom_spectra" / "my_star.txt"
    _write_spectrum_file(custom, fl=(111.0, 222.0))

    handler = _make_handler_for_init_star(tmp_path, spectrum_source=None, star_path=str(custom))

    init_star(handler)

    backup = tmp_path / "out" / "data" / "-1.sflux"
    arr = np.loadtxt(backup)
    assert np.allclose(arr[:, 1], np.array([111.0, 222.0]))


def test_init_star_phoenix_branch_uses_get_phoenix_modern_spectrum(tmp_path, monkeypatch):
    import proteus.star.wrapper as wrapper_mod
    from proteus.star.wrapper import init_star

    fake_mors = _install_fake_mors(monkeypatch)

    phoenix_modern = tmp_path / "stellar_spectra" / "PHOENIX" / "1AU" / "fake_phoenix.txt"
    _write_spectrum_file(phoenix_modern, fl=(7.0, 8.0))

    calls = {"n": 0, "track_type": None}

    def fake_get_phoenix_modern_spectrum(handler, stellar_track=None, age_yr=None):
        calls["n"] += 1
        calls["track_type"] = type(stellar_track).__name__ if stellar_track is not None else None
        return phoenix_modern

    monkeypatch.setattr(wrapper_mod, "get_phoenix_modern_spectrum", fake_get_phoenix_modern_spectrum)

    handler = _make_handler_for_init_star(tmp_path, spectrum_source="phoenix", tracks="baraffe", offline=True)
    init_star(handler)

    assert calls["n"] == 1
    assert calls["track_type"] == "FakeBaraffeTrack"

    backup = tmp_path / "out" / "data" / "-1.sflux"
    arr = np.loadtxt(backup)
    assert np.allclose(arr[:, 1], np.array([7.0, 8.0]))
    assert isinstance(handler.stellar_track, fake_mors.BaraffeTrack)
