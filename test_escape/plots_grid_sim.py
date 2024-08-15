import pandas as pd
import matplotlib.pyplot as plt
import os

path_to_csv = '/Users/emmapostolec/Documents/PHD/SCIENCE/CODES/PROTEUS/output/'
plot_dir = 'plot_comparison_grid'

# Create the directory for saving plots if it does not exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Directory names and corresponding labels
directory_labels = {
    'dummy_escape_rate_4e3': r'Dummy Escape : 4e3 [kg s$^{-1}$]',
    'dummy_escape_rate_4e5': r'Dummy Escape : 4e5 [kg s$^{-1}$]',
    'dummy_escape_rate_4e7': r'Dummy Escape : 4e7 [kg s$^{-1}$]',
    'dummy_escape_rate_4e9': r'Dummy Escape : 4e9 [kg s$^{-1}$]',
    'el_escape_eps_0.15': r'EL Escape : $\epsilon$ = 0.15',
    'el_escape_eps_0.30': r'EL Escape : $\epsilon$ = 0.30',
    'el_escape_eps_0.45': r'EL Escape : $\epsilon$ = 0.45',
    'el_escape_eps_0.60': r'EL Escape : $\epsilon$ = 0.60'
}

csv_file = 'runtime_helpfile.csv'

# Define a fixed list of colors in the specified order
color_list = ['red', 'orange', 'gold', 'green', 'deepskyblue', 'blue', 'blueviolet', 'magenta']

# Ensure we have enough colors by repeating the list if needed
color_list = (color_list * (len(directory_labels) // len(color_list) + 1))[:len(directory_labels)]

# Plot all datasets
plt.figure(figsize=(12, 8))
for i, (directory, label) in enumerate(directory_labels.items()):
    full_path = os.path.join(path_to_csv, directory, csv_file)
    
    if os.path.isfile(full_path):
        df = pd.read_csv(full_path, delimiter='\t')
        time_column = df['Time']
        escape_rate_column = df['esc_rate_total']
        
        # Determine line style based on directory name
        if 'dummy_escape_rate' in directory:
            linestyle = '--'  # Dashed line for dummy escape rates
        elif 'el_escape_eps' in directory:
            linestyle = '-'  # Solid line for EL escape rates
        
        plt.loglog(time_column, escape_rate_column, linestyle=linestyle, color=color_list[i], label=label)
    else:
        print(f"Warning: {full_path} does not exist.")

plt.xlabel('Time [years]')
plt.ylabel(r'Total Escape Rate [kg s$^{-1}$]')
plt.title('Comparison of All Escape Rates')
plt.legend()
plt.grid(alpha=0.5)
plt.savefig(os.path.join(plot_dir, 'comparison_all.png'), dpi=180)
plt.show()

# Plot dummy_escape_rate datasets
plt.figure(figsize=(12, 8))
for i, (directory, label) in enumerate(directory_labels.items()):
    if 'dummy_escape_rate' in directory:
        full_path = os.path.join(path_to_csv, directory, csv_file)
        
        if os.path.isfile(full_path):
            df = pd.read_csv(full_path, delimiter='\t')
            time_column = df['Time']
            escape_rate_column = df['esc_rate_total']
            
            plt.loglog(time_column, escape_rate_column, linestyle='--', color=color_list[i], label=label)
        else:
            print(f"Warning: {full_path} does not exist.")

plt.xlabel('Time [years]')
plt.ylabel(r'Total Escape Rate [kg s$^{-1}$]')
plt.title('Comparison of Dummy Escape Rates')
plt.legend()
plt.grid(alpha=0.5)
plt.savefig(os.path.join(plot_dir, 'comparison_dummy_escape.png'), dpi=180)
plt.show()

# Plot el_escape_eps datasets
path_to_csv = '/Users/emmapostolec/Documents/PHD/SCIENCE/CODES/PROTEUS/output/'
plot_dir = 'plot_comparison_grid'

# Create the directory for saving plots if it does not exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Directory names and corresponding labels (in reverse order)
directory_labels = {
    'el_escape_eps_0.60': r'$\epsilon$ = 0.60',
    'el_escape_eps_0.45': r'$\epsilon$ = 0.45',
    'el_escape_eps_0.30': r'$\epsilon$ = 0.30',
    'el_escape_eps_0.15': r'$\epsilon$ = 0.15'
}

csv_file = 'runtime_helpfile.csv'

# Define a fixed list of colors in the specified order
color_list = ['deepskyblue', 'blue', 'blueviolet', 'magenta']

# Define zoom regions (in reverse order)
zoom_ranges = [
    {'y': (1.6596e4, 1.6606e4), 'x': (1e3, 1e7)},
    {'y': (1.2447e4, 1.24545e4), 'x': (1e3, 1e7)},
    {'y': (8.298e3, 8.303e3), 'x': (1e3, 1e7)},
    {'y': (4.1490e3, 4.1515e3), 'x': (1e3, 1e7)}
]

# Adjust the figure size to fit the grid layout
fig = plt.figure(figsize=(12, 6))

# Create the main plot
main_ax = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=4)
for i, (directory, label) in enumerate(directory_labels.items()):
    full_path = os.path.join(path_to_csv, directory, csv_file)
    
    if os.path.isfile(full_path):
        df = pd.read_csv(full_path, delimiter='\t')
        time_column = df['Time']
        escape_rate_column = df['esc_rate_total']
        
        main_ax.loglog(time_column, escape_rate_column, linestyle='-', color=color_list[i], label=label)
    else:
        print(f"Warning: {full_path} does not exist.")

main_ax.set_xlabel('Time [years]')
main_ax.set_ylabel(r'Total Escape Rate [kg s$^{-1}$]')
main_ax.set_title('Comparison of EL Escape Rates')

# Position the legend at the "best" location
main_ax.legend(loc='best', fontsize=10)
main_ax.grid(alpha=0.5)

# Create the zoom panels in a 1x4 grid on the right side of the main plot
zoom_axs = []
zoom_axs.append(plt.subplot2grid((4, 4), (0, 3)))
zoom_axs.append(plt.subplot2grid((4, 4), (1, 3)))
zoom_axs.append(plt.subplot2grid((4, 4), (2, 3)))
zoom_axs.append(plt.subplot2grid((4, 4), (3, 3)))

# Plot each zoom panel
for i, (zoom_range, (directory, label)) in enumerate(zip(zoom_ranges, directory_labels.items())):
    zoom_ax = zoom_axs[i]
    
    # Plot the zoomed data on the zoom panel
    for j, (dir_name, lbl) in enumerate(directory_labels.items()):
        full_path = os.path.join(path_to_csv, dir_name, csv_file)
        
        if os.path.isfile(full_path):
            df = pd.read_csv(full_path, delimiter='\t')
            time_column = df['Time']
            escape_rate_column = df['esc_rate_total']
            
            zoom_ax.loglog(time_column, escape_rate_column, linestyle='-', color=color_list[j], label=lbl)
    
    zoom_ax.set_ylim(zoom_range['y'])
    zoom_ax.set_xlim(zoom_range['x'])
    
    # Label the y-axis only with min and max values
    y_min, y_max = zoom_range['y']
    zoom_ax.set_yticks([y_min, y_max])
    zoom_ax.set_yticklabels([f'{y_min:.2e}', f'{y_max:.2e}'])
    
    # Set x-axis labels to scientific notation
    zoom_ax.set_xticks([10**3, 10**4, 10**5, 10**6, 10**7])
    zoom_ax.set_xticklabels([r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$', r'$10^7$'])
    
    # Set zoom panel title
    zoom_ax.set_title(f'Zoom: {label}', fontsize=10)
    zoom_ax.grid(alpha=0.5)

# Adjust layout
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'comparison_el_escape_with_zoom_column_best_legend.png'), dpi=180)
plt.show()

# Plot each el_escape_eps dataset individually
for i, (directory, label) in enumerate(directory_labels.items()):
    if 'el_escape_eps' in directory:
        plt.figure(figsize=(12, 8))
        full_path = os.path.join(path_to_csv, directory, csv_file)
        
        if os.path.isfile(full_path):
            df = pd.read_csv(full_path, delimiter='\t')
            time_column = df['Time']
            escape_rate_column = df['esc_rate_total']
            
            plt.loglog(time_column, escape_rate_column, linestyle='-', color=color_list[i], label=label)
            plt.xlabel('Time [years]')
            plt.ylabel(r'Total Escape Rate [kg s$^{-1}$]')
            plt.title(f'{label}')
            plt.legend()
            plt.grid(alpha=0.5)
            plt.savefig(os.path.join(plot_dir, f'plot_{directory}.png'), dpi=180)
            plt.show()
        else:
            print(f"Warning: {full_path} does not exist.")
