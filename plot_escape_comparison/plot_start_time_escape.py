import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path_to_csv = '/Users/emmapostolec/Documents/PHD/SCIENCE/CODES/PROTEUS/output/'

# Directory names and corresponding labels
directory_labels = {
    'escape_start_100Myr': r'Time since star formation : 100 Myr',
    'escape_start_200Myr': r'Time since star formation : 200 Myr',
    'escape_start_400Myr': r'Time since star formation : 400 Myr',
    'escape_start_600Myr': r'Time since star formation : 600 Myr',
    'escape_start_800Myr': r'Time since star formation : 800 Myr',
    'escape_start_1000Myr': r'Time since star formation : 1000 Myr'
}

csv_file = 'runtime_helpfile.csv'

# Plot 
plt.figure(figsize=(9, 6))
for i, (directory, label) in enumerate(directory_labels.items()):
    full_path = os.path.join(path_to_csv, directory, csv_file)
    
    if os.path.isfile(full_path):
        df = pd.read_csv(full_path, delimiter='\t')
        time_column = df['Time']
        escape_rate_column = df['esc_rate_total']
        
        plt.plot(time_column, escape_rate_column, label=label)
    else:
        print(f"Warning: {full_path} does not exist.")

plt.xlabel('Time [years]')
plt.ylabel(r'Total Escape Rate [kg s$^{-1}$]')
plt.yscale('log')
# plt.xlim(left=1e6)  # Set x-axis to start at 
# plt.xlim(right=3.4e6)  # Set x-axis to stop at
plt.legend(loc='best')
plt.grid(alpha=0.5)
plt.savefig(os.path.join('escape_for_different_starting_time.png'), dpi=180)
plt.show()