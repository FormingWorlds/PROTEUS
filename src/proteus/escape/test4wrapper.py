from netCDF4 import Dataset
import numpy as np
# Replace 'your_file.nc' with your actual file name
nc_file = '/Users/emmapostolec/Documents/PHD/SCIENCE/CODES/PROTEUS/output/default/data/27785_atm.nc'

# Open the NetCDF file
dataset = Dataset(nc_file)

# Print the file information
print(np.len(dataset))

# List all variables
#print("Variables:")
#for var in dataset.variables:
#print(var)

# Close the dataset
dataset.close()
