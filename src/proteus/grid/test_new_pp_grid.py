from __future__ import annotations

from pathlib import Path

import post_processing_updated as pp

# Point this to your actual grid folder
grid_name = "escape_grid_1Msun"
grid_path = Path(f"/projects/p315557/Paper_1/DATA/Grids/{grid_name}")

data = pp.load_grid_cases(grid_path)
case_params = pp.get_tested_grid_parameters(data,grid_path)
pp.generate_summary_csv(data,case_params,grid_path,grid_name)
pp.generate_completed_summary_csv(data,case_params,grid_path,grid_name)
pp.generate_running_error_summary_csv(data,case_params,grid_path,grid_name)
