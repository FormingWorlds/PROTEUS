from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import pandas as pd
import post_processing_updated as pp

grid_name = "escape_grid_1Msun"
grid_path = Path(f"/projects/p315557/Paper_1/DATA/Grids/{grid_name}")

# grid_name= "toi561b_grid"
# grid_path = Path(f"/projects/p315557/TOI-561b/{grid_name}")

# grid_name= "escape_on_off_janus_agni_1Msun"
# grid_path = Path(f"/projects/p315557/Paper_1/DATA/Comparison_escape_on_off/{grid_name}")

data = pp.load_grid_cases(grid_path)
input_param_grid_per_case, tested_params_grid = pp.get_tested_grid_parameters(data,grid_path)
#pp.generate_summary_csv(data,case_params,grid_path,grid_name)
#pp.generate_completed_summary_csv(data,case_params,grid_path,grid_name)
#pp.generate_running_error_summary_csv(data,case_params,grid_path,grid_name)

#pp.plot_grid_status(data,grid_path,grid_name)
rows = []
for case_index, case in enumerate(data):
    row = {}

    # Add input parameters
    init_params = case['init_parameters']
    for k, v in init_params.items():
        row[k] = v

    # Add outputs (last timestep)
    df_out = case['output_values']
    if df_out is not None:
        last_row = df_out.iloc[-1].to_dict()
        row.update(last_row)

    # Add status
    row['status'] = case['status']

    rows.append(row)

df = pd.DataFrame(rows)
def flatten_nested_columns(df):
    flat_data = {}
    for col in df.columns:
        if isinstance(df[col].iloc[0], dict):
            # Expand each dictionary into separate columns
            nested_df = pd.json_normalize(df[col])
            nested_df.index = df.index
            for nested_col in nested_df.columns:
                flat_data[f"{col}.{nested_col}"] = nested_df[nested_col]
        else:
            flat_data[col] = df[col]
    return pd.DataFrame(flat_data)
df_flat = flatten_nested_columns(df)
grouped_data = pp.group_output_by_parameter(df_flat, tested_params_grid, ['T_surf'])


param_settings_grid = {
    "atmos_clim.module":          {"label": "Atmosphere module",                      "colormap": cm.Paired,    "log_scale": False},
    "orbit.semimajoraxis":        {"label": "a [AU]",                                 "colormap": cm.plasma,   "log_scale": False},
    "escape.zephyrus.efficiency": {"label": r"$\rm \epsilon$",                            "colormap": cm.spring,   "log_scale": False},
    "escape.zephyrus.Pxuv":       {"label": r"P$_{\rm XUV}$ [bar]",                       "colormap": cm.cividis,  "log_scale": True},
    "outgas.fO2_shift_IW":        {"label": r"$\rm \log_{10} fO_2 [IW]$",                "colormap": cm.coolwarm, "log_scale": False},
    "delivery.elements.CH_ratio": {"label": "C/H ratio",                              "colormap": cm.copper,   "log_scale": False},
    "delivery.elements.H_oceans": {"label": "[H] [oceans]",                           "colormap": cm.winter,   "log_scale": False}}
output_settings_grid = {
    'solidification_time': {"label": "Solidification [yr]",             "log_scale": True,  "scale": 1.0},
    'Phi_global':          {"label": "Melt fraction [%]",               "log_scale": False, "scale": 100.0},
    'T_surf':              {"label": r"T$_{\rm surf}$ [$10^3$ K]",                 "log_scale": False,  "scale": 1.0/1000.0},
    'P_surf':              {"label": r"P$_{\rm surf}$ [bar]",          "log_scale": True,  "scale": 1.0},
    'atm_kg_per_mol':      {"label": "MMW [g/mol]",                     "log_scale": False,  "scale": 1000.0},
    'esc_rate_total':      {"label": "Escape rate [kg/s]",              "log_scale": True,  "scale": 1.0}}

pp.ecdf_grid_plot(tested_params_grid, grouped_data, param_settings_grid, output_settings_grid, grid_path, grid_name)
