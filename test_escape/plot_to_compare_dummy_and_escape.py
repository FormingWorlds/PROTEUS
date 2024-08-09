import pandas as pd
import matplotlib.pyplot as plt

path_to_csv = '/Users/emmapostolec/Documents/PHD/SCIENCE/CODES/PROTEUS/output/earth_demo/'

# Load .csv files
escape_csv            = 'runtime_helpfile.csv'
df_escape             = pd.read_csv(path_to_csv+escape_csv, delimiter='\t')   # Zephyrus escape
time_column_e         = df_escape['Time']
escape_rate_column_e  = df_escape['esc_rate_total']

dummy_csv             = 'runtime_helpfile.csv'
df_dummy              = pd.read_csv(path_to_csv+dummy_csv, delimiter='\t')    # Dummy escape
time_column_d         = df_dummy['Time']
escape_rate_column_d  = df_dummy['esc_rate_total']

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(time_column_d, escape_rate_column_d, label='Dummy escape', color='orange')
plt.loglog(time_column_e, escape_rate_column_e, label='EL escape (Zephyrus)', color='steelblue')
plt.xlabel('Time [years]')
plt.ylabel('Total Escape Rate [kg s^-1]')
plt.legend()
plt.grid(alpha=0.5)
plt.savefig('comparison_dummy_zephyrus.png', dpi=180)