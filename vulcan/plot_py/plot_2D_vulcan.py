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


vul_data = 'output/4X-TPG-vxmax3e5-KzzE10.vul'
vul_data2 = 'output/photo-4X_thor_wasp33b_Kzz.vul'


# Setting the 2rd input argument as the species names to be plotted (separated by ,)
sp = sys.argv[1]
# Setting the 3th input argument as the output eps filename        
plot_name = sys.argv[2]

plot_dir = vulcan_cfg.plot_dir

# taking user input species and splitting into separate strings and then converting the list to a tuple
#plot_spec = tuple(plot_spec.split(','))
#nspec = len(plot_spec)

colors = ['c','b','g','r','m','y','k','orange','pink','grey','darkred','darkblue','salmon','chocolate','steelblue','plum','hotpink']
colors = ['red', 'purple', 'b', 'green', 'pink','grey','darkred','darkblue','salmon','chocolate','steelblue','plum','hotpink']

tex_labels = {'H':'H','H2':'H$_2$','O':'O','OH':'OH','H2O':'H$_2$O','CH':'CH','C':'C','CH2':'CH$_2$','CH3':'CH$_3$','CH4':'CH$_4$','HCO':'HCO','H2CO':'H$_2$CO', 'C4H2':'C$_4$H$_2$',\
'C2':'C$_2$','C2H2':'C$_2$H$_2$','C2H3':'C$_2$H$_3$','C2H':'C$_2$H','CO':'CO','CO2':'CO$_2$','He':'He','O2':'O$_2$','CH3OH':'CH$_3$OH','C2H4':'C$_2$H$_4$','C2H5':'C$_2$H$_5$','C2H6':'C$_2$H$_6$','CH3O': 'CH$_3$O'\
,'CH2OH':'CH$_2$OH', 'NH3':'NH$_3$'}


with open(vul_data, 'rb') as handle:
  data = pickle.load(handle)
with open(vul_data2, 'rb') as handle:
  data2 = pickle.load(handle)

color_index = 0
vulcan_spec = data['variable']['species']
#vulcan_spec2 = data2['variable']['species']

# for sp in plot_spec:
#     if color_index == len(colors): # when running out of colors
#         colors.append(tuple(np.random.rand(3)))
#     if sp in tex_labels: sp_lab = tex_labels[sp]
#     else: sp_lab = sp

k_list = ['Day', 'Evening', 'Night', 'Morning']
#k_list = ['day', 'night']
for k in range(data['atm']['pco'].shape[0]):
    
    plt.plot(data['variable']['ymix'][k][:,vulcan_spec.index(sp)], data['atm']['pco'][k]/1.e6, color=colors[k], label=k_list[k], alpha=0.7)
    plt.plot(data['variable']['y_ini'][k][:,vulcan_spec.index(sp)]/data['atm']['n_0'][k], data['atm']['pco'][k]/1.e6, color=colors[k], ls='--', alpha=0.7)
    
    #plt.plot(data2['variable']['ymix'][k][:,vulcan_spec.index(sp)], data2['atm']['pco'][k]/1.e6, color=colors[k], ls='--', lw=1.5, alpha=0.8)
    #if sp in data2['variable']['species']:
        #plt.plot(data2['variable']['ymix'][:,vulcan_spec2.index(sp)], data2['atm']['pco']/1.e6, color=colors[color_index], ls='--',lw=1.5)
    
plt.gca().set_xscale('log')       
plt.gca().set_yscale('log') 
plt.gca().invert_yaxis() 
#plt.xlim(xmin=1e-15)
plt.ylim((data['atm']['pco'][0][0]/1e6,data['atm']['pco'][0][-1]/1e6))
plt.legend(frameon=0, prop={'size':12}, loc='best')
plt.xlabel("Mixing Ratio")
plt.ylabel("Pressure (bar)")
plt.title(tex_labels[sp])
plt.savefig(plot_dir + plot_name + '.png')
plt.savefig(plot_dir + plot_name + '.eps')
if vulcan_cfg.use_PIL == True:
    plot = Image.open(plot_dir + plot_name + '.png')
    plot.show()
else: plt.show()

