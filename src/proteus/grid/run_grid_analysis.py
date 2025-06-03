# This script is used to post-process a grid of PROTEUS simulations.
# It extracts the output values from the simulation cases, saves them to a CSV file,
# and generates plots for the grid statuses based on the extracted data. It also generates
# ECDF single plots and a big grid plot for the input parameters vs extracted outputs.

# The users need to specify the path to the grid directory and the grid name. (see the example below)
# He also needs to specify the output columns to extract from the 'runtime_helpfile.csv' of each case and 
# update the related plotting variables accordingly. This can be done in the `run_grid_postprocessing` function. (see src/proteus/grid/postprocess_grid.py)

from post_processing_grid import run_grid_analyze

if __name__ == "__main__":
    run_grid_analyze(path_to_grid="/home2/p315557/PROTEUS/output/scratch/",
                            grid_name="escape_grid_habrok_7_params_1Msun",
                            update_csv=True)
