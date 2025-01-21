import toml
import subprocess
import pandas as pd

def run_proteus(config_file, output_file):
    """
    Run the simulator using a .toml configuration file and return the results as a DataFrame.

    Parameters:
        config_file (str): Path to the .toml configuration file.
        output_file (str): Path to the simulator's output .csv file.

    Returns:
        pd.DataFrame: Output data loaded into a Pandas DataFrame.
    """

    # Run the simulator as a subprocess
    try:
        subprocess.run(
            ["proteus", "start", "-c", config_file],
            check=True,
            stdout=subprocess.DEVNULL # to suppress terminal output
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Simulator failed with error: {e}")

    # Load the output CSV file into a Pandas DataFrame
    try:
        df = pd.read_csv(output_file, delimiter="\t")
    except FileNotFoundError:
        raise RuntimeError(f"Output file {output_file} not found. Check the simulator configuration.")

    out = df.iloc[-1][[    "z_obs",
                            "R_int",
                            "M_planet",
                            "transit_depth",
                            "bond_albedo",
                            "contrast_ratio"]]
    return out

def update_toml(config_file, updates, output_file=None):
    """
    Update values in a .toml configuration file.

    Parameters:
        config_file (str): Path to the existing .toml file.
        updates (dict): Dictionary of updates to apply, with support for nested keys (e.g., "section.key").
        output_file (str): Optional path to save the updated .toml file. If None, overwrites the input file.

    Returns:
        None
    """
    # Load the current configuration
    with open(config_file, 'r') as f:
        config = toml.load(f)

    # Apply updates
    for key, value in updates.items():
        keys = key.split('.')  # Support for nested keys with dot notation
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})  # Create nested dictionaries if they don't exist
        d[keys[-1]] = value

    # Save the updated configuration
    output_file = output_file or config_file  # Default to overwriting the input file
    with open(output_file, 'w') as f:
        toml.dump(config, f)

    # print(f"Configuration file updated and saved to {output_file}.")
