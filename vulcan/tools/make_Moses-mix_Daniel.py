import numpy as np 
import pickle
from scipy import interpolate

vul = '../output/rtol005-new2-HD189.vul'
with open(vul, 'rb') as handle:
  vul = pickle.load(handle)

# interpolating Moses's z from my TP
vulPz_fun = interpolate.interp1d(vul['atm']['pico'], vul['atm']['zco'], assume_sorted = False, bounds_error=False,fill_value=(vul['atm']['zco'][np.argmin(vul['atm']['pico'])], vul['atm']['zco'][np.argmax(vul['atm']['pico'])] )  )



# Read Moses 2011 for Tiso and const-Kzz 
JM, JM_labels = {}, {}
JM[0] = np.genfromtxt('../atm/JM/reorder0_HD189_wtday.txt', names=True)
for i in  range(1,10):
    JM[i] = np.genfromtxt('../atm/JM/reorder'+str(i)+'_HD189_wtday.txt',names=True)
    JM_labels[i] = np.genfromtxt('../atm/JM/reorder'+str(i)+'_HD189_wtday.txt', dtype=str)[0]



output = open('../output/HD189-Moses.txt', "w")
# C2H2, CH4, CO, CO2, H2, H2O, HCN, He, NH3

JM['mix'] = {}
sp_list = ['C2H2', 'CH4', 'CO', 'CO2', 'H2', 'H2O', 'HCN', 'NH3', 'N2', 'HE']
for sp in sp_list:
    for i in  range(1,10):
        if sp in JM_labels[i]:
            JM['mix'][sp] = JM[i][sp]/JM[0]['DENSITY']

ost = '{:<8s}'.format('(dyn/cm2)')  + '{:>9s}'.format('(K)') + '{:>9s}'.format('(cm)') +'\n'
ost += '{:<8s}'.format('Pressure')  + '{:>9s}'.format('Temp')+ '{:>9s}'.format('Hight')+ '{:>10s}'.format('CH4') +\
'{:>10s}'.format('CO') + '{:>10s}'.format('H2O')+ '{:>10s}'.format('CO2') + '{:>10s}'.format('H2') + '{:>10s}'.format('He') + \
'{:>10s}'.format('C2H2') + '{:>10s}'.format('N2') + '{:>10s}'.format('NH3') + '{:>10s}'.format('HCN') +'\n'

z = 0 
for n, p in enumerate(JM[0]['PRESSURE']*1.e3):
    # if not n==len(JM[0]['PRESSURE'])-1:
    #     rho = JM[0]['PRESSURE'][n]*1.e3/JM[0]['TEMPERATURE'][n]/8.314E7  /2.4
    #     dz =  (JM[0]['PRESSURE'][n]-JM[0]['PRESSURE'][n+1])*1.e3/ (rho*2140.)
    # z += dz
    
    ost += '{:<8.3E}'.format(p)  + '{:>8.1f}'.format(JM[0]['TEMPERATURE'][n])  + '{:>10.2E}'.format(float(vulPz_fun(p)))  +\
    '{:>10.2E}'.format(JM['mix']['CH4'][n])  +'{:>10.2E}'.format(JM['mix']['CO'][n]) +'{:>10.2E}'.format(JM['mix']['H2O'][n]) +\
    '{:>10.2E}'.format(JM['mix']['CO2'][n]) + '{:>10.2E}'.format(JM['mix']['H2'][n]) +'{:>10.2E}'.format(JM['mix']['HE'][n]) +\
    '{:>10.2E}'.format(JM['mix']['C2H2'][n]) + '{:>10.2E}'.format(JM['mix']['N2'][n]) +'{:>10.2E}'.format(JM['mix']['NH3'][n]) \
    +'{:>10.2E}'.format(JM['mix']['HCN'][n]) + '\n'

output.write(ost)
output.close()

