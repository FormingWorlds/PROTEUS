import os

dir_socrates    = "/home/x_ryabo/PROTEUS/AEOLUS/rad_trans/socrates_code/"

# File name of SOCRATES spectral file to be generated
file_name = "Reach"

# File name of bash execution script to be written
exec_file_name = "sp_exec_"+file_name+".sh"

# Open file to produce bash script
f = open(exec_file_name, "w+")

f.write("echo '###############   H2O cloud  ###############'"+"\n")
f.write('Cscatter_average -s '+file_name+' -P 1 -t -p 250 -f 5 fit_lw_drop5 mon_lw_drop5 1.e3 '+dir_socrates+'data/cloud/scatter_drop_type5'+ '\n')
# Add to spec file
f.write('prep_spec <<EOF'+ '\n')
# Select spectral file
f.write(file_name + '\n')
# Append
f.write('a'+ '\n')
# Select block 10 (droplet parameters)
f.write('10'+ '\n')
# Agree to overwrite (uncomment if needed)
#f.write('y'+ '\n')
# Enter the number for the type of droplets for which data are to be provided
f.write('5'+ '\n')
# Enter the name of the file containing the fitted parameters for the droplets
f.write('fit_lw_drop5'+ '\n')
# Enter the range of validity of the parametrization
f.write('1.50000E-06 5.00000E-05'+ '\n') # Pade fits, function of droplet effective radius. It is a droplet radius in metres. One should use the values from the ParametrisedOpticalProperties tutorial if using the scatter_drop_type5 file.
f.write('-1'+ '\n')
f.write('EOF'+ '\n')

# Close
f.close()
os.chmod(exec_file_name,0o777)

#######################

# Run script
os.system('./'+exec_file_name)


