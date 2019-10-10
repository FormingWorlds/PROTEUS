import numpy as np 
import pickle

vul = '../output/new-H2O-HCN-NO2-HD189.vul'
#vul_east = 'output/wasp43b_ter_east_CtoO1.vul'

with open(vul, 'rb') as handle:
  vul = pickle.load(handle)
# with open(vul_east, 'rb') as handle:
#   vul_east = pickle.load(handle)
  
species = vul['variable']['species'] 

  
dz_vul = vul['atm']['zco']
# zz = np.zeros(len(dz_vul))
# for n,dz in enumerate(dz_vul):
#     if n==1: zz[n] = dz
#     elif n>1:  zz[n] = zz[n-1] + dz


#thor_day = np.genfromtxt('output/thor_ter_east.txt',names=True, dtype=None, skip_header=1)
#thor_west = np.genfromtxt('output/thor_ter_west.txt',names=True, dtype=None, skip_header=1)
#thor_east = np.genfromtxt('output/thor_ter_east.txt',names=True, dtype=None, skip_header=1)

#fc = np.genfromtxt('fastchem_vulcan/output/vulcan_EQ_west.dat', names=True, dtype=None, skip_header=0)   
#fc = np.genfromtxt('fastchem_vulcan/output/vulcan_EQ_east.dat', names=True, dtype=None, skip_header=0)   


output = open('../output/HD189-vulcan.txt', "w")
# C2H2, CH4, CO, CO2, H2, H2O, HCN, He, NH3

ost = '{:<8s}'.format('(dyn/cm2)')  + '{:>9s}'.format('(K)') + '{:>9s}'.format('(cm)') +'\n'
ost += '{:<8s}'.format('Pressure')  + '{:>9s}'.format('Temp')+ '{:>9s}'.format('Hight')+ '{:>10s}'.format('CH4') +\
'{:>10s}'.format('CO') + '{:>10s}'.format('H2O')+ '{:>10s}'.format('CO2') + '{:>10s}'.format('H2') + '{:>10s}'.format('He') + \
'{:>10s}'.format('C2H2') + '{:>10s}'.format('N2') + '{:>10s}'.format('NH3') + '{:>10s}'.format('HCN') +'\n'
 
for n, p in enumerate(vul['atm']['pco']):
    ost += '{:<8.3E}'.format(p)  + '{:>8.1f}'.format(vul['atm']['Tco'][n])  + '{:>10.2E}'.format(vul['atm']['zco'][n])  +\
    '{:>10.2E}'.format(vul['variable']['ymix'][n,species.index('CH4')])  +'{:>10.2E}'.format(vul['variable']['ymix'][n,species.index('CO')]) +'{:>10.2E}'.format(vul['variable']['ymix'][n,species.index('H2O')]) +\
    '{:>10.2E}'.format(vul['variable']['ymix'][n,species.index('CO2')]) + '{:>10.2E}'.format(vul['variable']['ymix'][n,species.index('H2')]) +'{:>10.2E}'.format(vul['variable']['ymix'][n,species.index('He')]) +\
    '{:>10.2E}'.format(vul['variable']['ymix'][n,species.index('C2H2')]) + '{:>10.2E}'.format(vul['variable']['ymix'][n,species.index('N2')]) +'{:>10.2E}'.format(vul['variable']['ymix'][n,species.index('NH3')]) \
    +'{:>10.2E}'.format(vul['variable']['ymix'][n,species.index('HCN')]) + '\n'

output.write(ost)
output.close()

