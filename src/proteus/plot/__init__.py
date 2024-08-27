from __future__ import annotations

from proteus.plot.cpl_global import plot_global_entry
from proteus.plot.cpl_interior import plot_interior_entry

plot_dispatch = {
    'interior': plot_interior_entry,
    'global': plot_global_entry,
}
