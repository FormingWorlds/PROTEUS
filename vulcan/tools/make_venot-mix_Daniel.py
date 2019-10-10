import numpy as np 
import pickle
from scipy import interpolate

vul = '../output/new-H2O-HCN-NO2-HD189.vul'
with open(vul, 'rb') as handle:
  vul = pickle.load(handle)
  
venot_HD189 = np.genfromtxt('../output/venot/venot_HD189_steady.dat',names=True)


# interpolating venot's T from my TP
vulTP_fun = interpolate.interp1d(vul['atm']['pco'], vul['atm']['Tco'], assume_sorted = False, bounds_error=False,fill_value=(vul['atm']['Tco'][np.argmin(vul['atm']['pco'])], vul['atm']['Tco'][np.argmax(vul['atm']['pco'])] )  )
             

output = open('../output/HD189-Venot.txt', "w")
# C2H2, CH4, CO, CO2, H2, H2O, HCN, He, NH3

ost = '{:<8s}'.format('(dyn/cm2)')  + '{:>9s}'.format('(K)') + '{:>9s}'.format('(cm)') +'\n'
ost += '{:<8s}'.format('Pressure')  + '{:>9s}'.format('Temp')+ '{:>9s}'.format('Hight')+ '{:>10s}'.format('CH4') +\
'{:>10s}'.format('CO') + '{:>10s}'.format('H2O')+ '{:>10s}'.format('CO2') + '{:>10s}'.format('H2') + '{:>10s}'.format('He') + \
'{:>10s}'.format('C2H2') + '{:>10s}'.format('N2') + '{:>10s}'.format('NH3') + '{:>10s}'.format('HCN') +'\n'
 
for n, p in enumerate(venot_HD189['P']*1.e3):
    ost += '{:<8.3E}'.format(p)  + '{:>8.1f}'.format(float(vulTP_fun(p)))  + '{:>10.2E}'.format(venot_HD189['z'][n]*1e5 - venot_HD189['z'][0]*1e5)  +\
    '{:>10.2E}'.format(venot_HD189['CH4'][n])  +'{:>10.2E}'.format(venot_HD189['CO'][n]) +'{:>10.2E}'.format(venot_HD189['H2O'][n]) +\
    '{:>10.2E}'.format(venot_HD189['CO2'][n]) + '{:>10.2E}'.format(venot_HD189['H2'][n]) +'{:>10.2E}'.format(venot_HD189['He'][n]) +\
    '{:>10.2E}'.format(venot_HD189['C2H2'][n]) + '{:>10.2E}'.format(venot_HD189['N2'][n]) +'{:>10.2E}'.format(venot_HD189['NH3'][n]) \
    +'{:>10.2E}'.format(venot_HD189['HCN'][n]) + '\n'

output.write(ost)
output.close()

