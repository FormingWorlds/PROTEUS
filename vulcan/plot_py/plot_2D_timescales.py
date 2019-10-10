import sys
sys.path.insert(0, '../') # including the upper level of directory for the path of modules

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.legend as lg
import vulcan_cfg
try: from PIL import Image
except ImportError: 
    try: import Image
    except: vulcan_cfg.use_PIL = False
import os, sys
import pickle


vul_data0 = '../output/4X-TPG-vxmax1e3-KzzE10.vul'
vul_data1 = '../output/4X-TPG-vxmax1e4-KzzE10.vul'
vul_data2 = '../output/4X-TPG-vxmax1e5-KzzE10.vul'
vul_data3 = '../output/4X-TPG-vxmax3e5-KzzE10.vul'
vul_data4 = '../output/4X-TPG-vxmax5e5-KzzE10.vul'


num_runs = 5
vx_list = [10,100,1000,3000,5000]

# Setting the 2rd input argument as the species names to be plotted (separated by ,)
#sp = sys.argv[1]
# Setting the 3th input argument as the output eps filename        
plot_name = sys.argv[1]

plot_dir = '../' + vulcan_cfg.plot_dir

# taking user input species and splitting into separate strings and then converting the list to a tuple
#plot_spec = tuple(plot_spec.split(','))
#nspec = len(plot_spec)

#colors = ['c','b','g','r','m','y','k','orange','pink','grey','darkred','darkblue','salmon','chocolate','steelblue','plum','hotpink']
colors = ['red', 'purple', 'b', 'green', 'pink','grey','darkred','darkblue','salmon','chocolate','steelblue','plum','hotpink']


data, mix = {}, {}
for index in range(num_runs):
    with open(eval('vul_data' + str(index)), 'rb') as handle:
        data[index] = pickle.load(handle)   

# with open(vul_data2, 'rb') as handle:
#   data2 = pickle.load(handle)

color_index = 0
#vulcan_spec = data['variable']['species']
#vulcan_spec2 = data2['variable']['species']


plt.figure('timescale') 

# plotting tau_z
for k in range(data[0]['atm']['pco'].shape[0]):
    Hpi = 0.5*(data[0]['atm']['Hp'][k] + np.roll(data[0]['atm']['Hp'][k],-1))
    Hpi = Hpi[:-1]
    plt.plot( Hpi**2. / data[0]['atm']['Kzz'][0] , data[0]['atm']['pico'][k][1:-1]/1.e6, color=colors[k], label='tau_v' + str(k))


# plotting tau_h
for index in range(num_runs): 
    plt.plot(data[index]['atm']['dx'] / data[index]['atm']['vx'][0], data[index]['atm']['pco'][0]/1.e6, color = plt.cm.Greys(0.1+0.9/num_runs*index), label='tau_h_'+str(vx_list[index]))





    
    
    #plt.plot(data2['variable']['ymix'][k][:,vulcan_spec.index(sp)], data2['atm']['pco'][k]/1.e6, color=colors[k], ls='--', lw=1.5, alpha=0.8)
    #if sp in data2['variable']['species']:
        #plt.plot(data2['variable']['ymix'][:,vulcan_spec2.index(sp)], data2['atm']['pco']/1.e6, color=colors[color_index], ls='--',lw=1.5)
    
plt.gca().set_xscale('log')       
plt.gca().set_yscale('log') 
plt.gca().invert_yaxis() 
#plt.xlim(xmin=0.)
plt.ylim((data[0]['atm']['pco'][0][0]/1e6,data[0]['atm']['pco'][0][-1]/1e6))
plt.legend(frameon=0, prop={'size':12}, loc='best')
plt.xlabel("Timescales (s)")
#plt.xlabel('sin', color='r')
plt.ylabel("Pressure (bar)")
#plt.title(tex_labels[sp])
plt.savefig(plot_dir + plot_name + '.png')
plt.savefig(plot_dir + plot_name + '.eps')
if vulcan_cfg.use_PIL == True:
    plot = Image.open(plot_dir + plot_name + '.png')
    plot.show()
else: plt.show()

