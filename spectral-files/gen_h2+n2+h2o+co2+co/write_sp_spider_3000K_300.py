import os
import numpy as np

# Remove junk files generated by previous runs
os.system('rm -rf write_spec_file')
os.system('rm -rf sp_spider')
os.system('rm -rf sp_spider_k')
os.system('rm -rf *_lbl')
os.system('rm -rf *_lbl.nc')
os.system('rm -rf *_m')
os.system('rm -rf *_o')
os.system('rm -rf *_o.nc')
os.system('rm -rf *_l')

# Open file to produce bash script
f= open("write_spec_file","w+")

# Write skeleton spectral file using prep_spec utility
f.write('prep_spec <<EOF'+ '\n')
f.write('sp_spider'+ '\n')

# Set number of bands
f.write('300'+ '\n')
# Set number of absorbers
f.write('5'+ '\n')
# Set absorber ids (see Socrates userguide)
f.write('1'+ '\n')
f.write('2'+ '\n')
f.write('5'+ '\n')
f.write('13'+ '\n')
f.write('23'+ '\n')
# Set number of continua
f.write('3'+ '\n')
f.write('23'+ '\n')
f.write('23'+ '\n')
f.write('13'+ '\n')
f.write('13'+ '\n')
f.write('13'+ '\n')
f.write('23'+ '\n')
# Set number of aerosols
f.write('0'+ '\n')
# Set band units (c for inverse cm)
f.write('c'+ '\n')

# Set band edges
bands = np.concatenate((np.arange(0,3000,20),np.arange(3000,9000,50),np.arange(9000,24500,500)))
bands[0] = 1.0

# Write band edges one by one
f.write(str(bands[0])+'\n')
for band in bands[1:-1]:
	f.write(str(band)+'\n')
	f.write(str(band)+'\n')
f.write(str(bands[-1])+'\n')

# Set absorbers in each band
for band in bands[:-1]:
	f.write('1 2 5'+ '\n')

# Set continua in each band
for band in bands[:-1]:
	f.write('1 2 3'+ '\n')

# Exclude no bands
f.write('n'+ '\n')

# Close prep_spec
f.write('-1'+ '\n')
f.write('EOF'+ '\n')

#####################

# Write correlated-k coefficients using Ccorr_k utility, with suffixes:
# -F File with grid of pressure-temperature values
# -R Range of bands for the calculation (default to all)
# -c Cutoff in inverse cm for each band
# -i Frequency increment for integration in m-1
# -l Maximum absorption path in kg/m2
# -t Tolerance, i.e. maximum RMS error in correlated-k assumption
# -k Adjust for use with CKD continuum (optional)
# -s Spectral file
# +p Planckian weighting
# -lk Use lookup table for pressure/temperature scaling
# -o, -m, -L Output files
# -np Number of processors

# CO
f.write('Ccorr_k -F pt48 -D 05_HITEMP2019.par -R 1 300 -c 2500.0 -i 1.0 -l 5 1.0e1 -t 1.0e-2 -k -s sp_spider +p -lk -o co_o -m co_m -L co_lbl.nc -np 16'+ '\n')

# Add to spec file with prep_spec
f.write('prep_spec <<EOF'+ '\n')
# Select spectral file
f.write('sp_spider'+ '\n')
# Append
f.write('a'+ '\n')
# Select block 5 (absorption data)
f.write('5'+ '\n')
# Select data
f.write('co_o'+ '\n')
f.write('-1'+ '\n')
f.write('EOF'+ '\n')

# N2H2
f.write('Ccorr_k -F pt_cont -CIA N2-H2_2011.cia -R 1 300 -i 1.0 -ct 23 13 1.0e2 -t 1.0e-2 -s ' + 'sp_spider' + ' +p -lk -o n2h2_o -m n2h2_m -L n2h2_lbl.nc' + '\n')

# Add to spec file with prep_spec
f.write('prep_spec <<EOF'+ '\n')
# Select spectral file
f.write('sp_spider'+ '\n')
# Append
f.write('a'+ '\n')
# Select block 19 (CIA data)
f.write('19'+ '\n')
# Select data
f.write('n2h2_o'+ '\n')
f.write('-1'+ '\n')
f.write('EOF'+ '\n')

# N2N2
f.write('Ccorr_k -F pt_cont -CIA N2-N2_2018.cia -R 1 300 -i 1.0 -ct 13 13 1.0e2 -t 1.0e-2 -s ' + 'sp_spider' + ' +p -lk -o n2n2_o -m n2n2_m -L n2n2_lbl.nc' + '\n')

# Add to spec file with prep_spec
f.write('prep_spec <<EOF'+ '\n')
# Select spectral file
f.write('sp_spider'+ '\n')
# Append
f.write('a'+ '\n')
# Select block 19 (CIA data)
f.write('19'+ '\n')
# Select data
f.write('n2n2_o'+ '\n')
f.write('-1'+ '\n')
f.write('EOF'+ '\n')

# H2H2
f.write('Ccorr_k -F pt_cont -CIA H2-H2_2011.cia -R 1 300 -i 1.0 -ct 23 23 1.0e2 -t 1.0e-2 -s ' + 'sp_spider' + ' +p -lk -o h2_o -m h2_m -L h2_lbl.nc' + '\n')

# Add to spec file with prep_spec
f.write('prep_spec <<EOF'+ '\n')
# Select spectral file
f.write('sp_spider'+ '\n')
# Append
f.write('a'+ '\n')
# Select block 19 (CIA data)
f.write('19'+ '\n')
# Select data
f.write('h2_o'+ '\n')
f.write('-1'+ '\n')
f.write('EOF'+ '\n')

# CO2
f.write('Ccorr_k -F pt48 -D co2_data.par -R 1 300 -c 2500.0 -i 1.0 -l 2 1.0e1 -t 1.0e-2 -k -s sp_spider +p -lk -o co2_o -m co2_m -L co2_lbl.nc -np 16'+ '\n')

# Add to spec file with prep_spec
f.write('prep_spec <<EOF'+ '\n')
# Select spectral file
f.write('sp_spider'+ '\n')
# Append
f.write('a'+ '\n')
# Select block 5 (absorption data)
f.write('5'+ '\n')
# Select data
f.write('co2_o'+ '\n')
f.write('-1'+ '\n')
f.write('EOF'+ '\n')



# H2O
f.write('Ccorr_k -F pt48 -D h2o_data.par -R 1 300 -c 2500.0 -i 1.0 -l 1 1.0e1 -t 1.0e-2 -k -s sp_spider +p -lk -o h2o_o -m h2o_m -L h2o_lbl.nc -np 16'+ '\n')

# Add to spec file
f.write('prep_spec <<EOF'+ '\n')
# Select spectral file
f.write('sp_spider'+ '\n')
# Append
f.write('a'+ '\n')
# Select block 5 (absorption data)
f.write('5'+ '\n')
# Agree to overwrite
f.write('y'+ '\n')
# Select data
f.write('h2o_o'+ '\n')

#####################

# Thermal source function
f.write('6'+ '\n')
f.write('n'+ '\n')
# Select table fit
f.write('T'+ '\n')
# Temperature range in K
f.write('100 4000'+ '\n')
# Number of points
f.write('100'+ '\n')

# Solar spectrum
f.write('2'+ '\n')
# No filter function
f.write('n'+ '\n')
# Path to spectrum data
f.write('/Users/markhammond/Work/Projects/1D-RC-SOC/socrates/socrates_main/data/solar/kurucz_95'+ '\n')
# Assign flux outside data range to bands on edge
f.write('y'+ '\n')

# Exit
f.write('-1'+ '\n')
f.write('EOF'+ '\n')

# Close
f.close()
os.chmod('write_spec_file',0o777)

#######################

# Remove junk files
os.system('rm -rf *_l')
os.system('rm -rf *_lm')
os.system('rm -rf *_lbl.nc')
os.system('rm -rf *_map.nc')

# Run script
os.system('./write_spec_file')
