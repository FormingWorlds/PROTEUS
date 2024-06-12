from utils.modules_ext import *



def rename_line(file_path, old_line, new_line):
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the index of the line to be renamed
    try:
        index = lines.index(old_line + '\n')
    except ValueError:
        print("Line not found in the file.")
        return

    # Replace the old line with the new one
    lines[index] = new_line + '\n'

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)





def katz2003(X_h20,directory,file_name):
    
    gamma=0.75
    K=43
    data = np.loadtxt('utils/Entropy-Temperature_interpolation.dat') 
    deltaT=K*pow(X_h20, gamma)
    T_new=data[:,0]-deltaT
    solid=np.genfromtxt('SPIDER/lookup_data/1TPa-dK09-elec-free/temperature_solid_test.dat')
    scaling_solid=solid[0]
    temp_solid=solid[1:]*scaling_solid
    da=temp_solid[0:]

    def conversion_to_entropy(data_solid,pressure_solid,temperature_interpolated):
        idx_entropy=np.where(np.isclose(data_solid[:,0]/1e9, pressure_solid, atol=3)& np.isclose(data_solid[:,2], np.round(temperature_interpolated), atol=15))
        return np.mean(data_solid[:,1][idx_entropy])

    E_new = list(map(lambda P, T: np.round(conversion_to_entropy(da, P, T)), data[:,1], T_new))

    E_new=np.array(E_new)
    nan_mask = ~np.isnan(E_new)
    E_new=E_new[nan_mask]
    P2_0=data[:,1][nan_mask]

    # Number of points to consider for slope calculation
    n_points = 10

    # Calculate the slope using the last n_points
    slope = (P2_0[-1] - P2_0[-n_points]) / (E_new[-1] - E_new[-n_points])

    # Extrapolate the curve
    extrapolated_x = np.linspace(E_new[-1], 2473.836256276209, 5)  
    extrapolated_y = P2_0[-1] + slope * (extrapolated_x - E_new[-1])

    entropy=np.concatenate((E_new, extrapolated_x))
    pressure=np.concatenate((P2_0, extrapolated_y))

    header='# 5'+' '+ f'{len(entropy)}' '\n# (Pressure, Entropy, Quantity) After re-computation  due to water in melt following Katz+2003 \n# column * scaling factor should be SI units \n# scaling factors (constant) for each column given on line below \n#1000000000.0 4824266.84604467'
    
    saved_file=np.savetxt(directory+file_name, X=np.array([pressure,entropy/4824266.84604467]).T,
            header=header,
            fmt='%.10e', delimiter='\t', comments='')
    
    return saved_file

