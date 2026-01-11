"""
Smoke test for JANUS-Interior coupling.

Verifies that JANUS atmosphere module can successfully couple with
dummy interior module and run for at last one timestep (binary execution).
"""

from __future__ import annotations

import pytest

from proteus import Proteus


@pytest.mark.janus
@pytest.mark.smoke
@pytest.mark.skip(reason='JANUS/SOCRATES runtime instability (hangs)')
def test_smoke_janus_dummy_coupling(tmp_path):
    """
    Test JANUS (atmos) + Dummy (interior) coupling (1 step).

    This test creates a minimal configuration file on the fly and runs
    PROTEUS for 1 iteration to verify that the JANUS Python-C bridge
    initializes correctly and exchanges fluxes with the interior.
    """

    # 1. Define minimal TOML content
    toml_content = """
# Minimal Smoke Test Config
version = "2.0"

[params]
    [params.out]
        path = "smoke_janus"
        logging = "INFO"
    [params.dt]
        initial = 1e2
        minimum = 1e1
        maximum = 1e3
        method = "maximum"
        [params.dt.proportional]
            propconst = 1.0
        [params.dt.adaptive]
            atol = 0.1
            rtol = 0.1
    [params.stop]
        strict = false
        [params.stop.iters]
            enabled = true
            minimum = 1
            maximum = 2
        [params.stop.time]
            enabled = false
        [params.stop.solid]
            enabled = false
        [params.stop.radeqm]
            enabled = false
        [params.stop.escape]
            enabled = false

[star]
    module = "dummy"
    mass = 1.0
    age_ini = 0.1
    [star.dummy]
        radius = 1.0
        Teff = 5772.0

[orbit]
    module = "dummy"
    semimajoraxis = 1.0
    eccentricity = 0.0
    zenith_angle = 48.2
    s0_factor = 0.25
    evolve = false
    [orbit.dummy]
        H_tide = 0.0
        Phi_tide = "<0.3"

[struct]
    mass_tot = 1.0
    corefrac = 0.5
    core_density = 8000.0
    core_heatcap = 1000.0

[interior]
    module = "dummy"
    grain_size = 0.01
    F_initial = 100.0
    radiogenic_heat = false
    tidal_heat = false
    rheo_phi_loc = 0.4
    rheo_phi_wid = 0.1
    bulk_modulus = 2e11
    melting_dir = "Monteux-600"
    [interior.dummy]
        ini_tmagma = 2000.0

[atmos_clim]
    module = "janus"
    prevent_warming = false
    surface_d = 0.01
    surface_k = 2.0
    cloud_enabled = false
    cloud_alpha = 0.0
    surf_state = "fixed"
    surf_greyalbedo = 0.1
    albedo_pl = 0.0
    rayleigh = false
    tmp_minimum = 10.0
    tmp_maximum = 5000.0

    [atmos_clim.janus]
        p_top = 1e-4
        p_obs = 1e-3
        spectral_group = "Frostflow"
        spectral_bands = 16
        F_atm_bc = 0
        num_levels = 30
        tropopause = "skin"
        overlap_method = "ro"

[outgas]
    module = "calliope"
    fO2_shift_IW = 0
    [outgas.calliope]
        T_floor = 500.0
        include_H2O = true
        include_CO2 = true
        include_N2 = true
        include_S2 = true
        include_SO2 = false
        include_H2S = false
        include_NH3 = false
        include_H2 = false
        include_CH4 = false
        include_CO = false

[delivery]
    module = "none"
    initial = "volatiles"
    radio_tref = 4.5
    radio_K = 0.0
    radio_U = 0.0
    radio_Th = 0.0
    [delivery.volatiles]
        H2O = 100.0
        CO2 = 10.0
        N2 = 0.0
        S2 = 0.0
        SO2 = 0.0
        H2S = 0.0
        NH3 = 0.0
        H2 = 0.0
        CH4 = 0.0
        CO = 0.0

[escape]
    module = "dummy"
    reservoir = "bulk"
    [escape.dummy]
        rate = 1000.0

[observe]
    synthesis = "none"

[atmos_chem]
    module = "none"
"""
    # 2. Write config
    cfg_file = tmp_path / 'smoke_janus.toml'
    cfg_file.write_text(toml_content)

    # 3. Initialize Proteus
    runner = Proteus(config_path=str(cfg_file))

    # Override output directory to use tmp_path to avoid cluttering real output/
    runner.directories['output'] = str(tmp_path / 'output')
    runner.directories['output/data'] = str(tmp_path / 'output/data')
    runner.directories['output/plots'] = str(tmp_path / 'output/plots')

    # 4. Run (offline mode)
    runner.start(offline=True)

    # 5. Verify success
    # Check if status file indicates completion or at least loop > 0
    # Since we set max iters = 1, it should finish.
    assert (tmp_path / 'output' / 'status').exists()
