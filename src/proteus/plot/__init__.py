from __future__ import annotations

from proteus.plot.cpl_atmosphere import plot_atmosphere_entry
from proteus.plot.cpl_atmosphere_cbar import plot_atmosphere_cbar_entry
from proteus.plot.cpl_bolometry import plot_bolometry_entry
from proteus.plot.cpl_emission import plot_emission_entry
from proteus.plot.cpl_escape import plot_escape_entry
from proteus.plot.cpl_fluxes_atmosphere import plot_fluxes_atmosphere_entry
from proteus.plot.cpl_fluxes_global import plot_fluxes_global_entry
from proteus.plot.cpl_global import plot_global_entry
from proteus.plot.cpl_interior import plot_interior_entry
from proteus.plot.cpl_interior_cmesh import plot_interior_cmesh_entry
from proteus.plot.cpl_population import plot_population_entry
from proteus.plot.cpl_sflux import plot_sflux_entry
from proteus.plot.cpl_sflux_cross import plot_sflux_cross_entry
from proteus.plot.cpl_spectra import plot_spectra_entry
from proteus.plot.cpl_structure import plot_structure_entry

# from proteus.plot.cpl_offchem_grid_cross import plot_offchem_grid_cross_entry
# from proteus.plot.cpl_offchem_species import plot_offchem_species_entry
# from proteus.plot.cpl_offchem_time import plot_offchem_time_entry
# from proteus.plot.cpl_offchem_year import plot_offchem_year_entry

plot_dispatch = {
    'atmosphere':           plot_atmosphere_entry,
    'atmosphere_cbar':      plot_atmosphere_cbar_entry,
    'escape':               plot_escape_entry,
    'fluxes_atmosphere':    plot_fluxes_atmosphere_entry,
    'fluxes_global':        plot_fluxes_global_entry,
    'global':               plot_global_entry,
    'interior':             plot_interior_entry,
    'interior_cmesh':       plot_interior_cmesh_entry,
    'bolometry':            plot_bolometry_entry,
    'spectra':              plot_spectra_entry,
    'sflux':                plot_sflux_entry,
    'sflux_cross':          plot_sflux_cross_entry,
    'structure':            plot_structure_entry,
    'emission':             plot_emission_entry,
    'population':           plot_population_entry,
    # 'offchem_grid_cross':   plot_offchem_grid_cross_entry,
    # 'offchem_species':      plot_offchem_species_entry,
    # 'offchem_time':         plot_offchem_time_entry,
    # 'offchem_year':         plot_offchem_year_entry,
}
