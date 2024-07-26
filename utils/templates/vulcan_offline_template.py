# ============================================================================= 
# Configuration file of VULCAN used as template in PROTEUS
# ============================================================================= 


# ====== Setting up paths and filenames for the input and output files  ======
# input:
use_lowT_limit_rates = False
gibbs_text = 'thermo/gibbs_text.txt' # (all the nasa9 files must be placed in the folder: thermo/NASA9/)
cross_folder = 'thermo/photo_cross/'
com_file = 'thermo/all_compose.txt'
top_BC_flux_file = 'atm/' # the file for the top boundary conditions
bot_BC_flux_file = 'atm/' # the file for the lower boundary conditions
# output:
output_dir = 'output/'
plot_dir = 'plot/'
movie_dir = 'plot/movie/'

# ====== Setting up photochemistry ======
use_photo = True
T_cross_sp = [] # warning: slower start! available atm: 'CO2','H2O','NH3', 'SH','H2S','SO2', 'S2', 'COS', 'CS2'

edd = 0.5 # the Eddington coefficient 
dbin1 = 0.1  # the uniform bin width < dbin_12trans (nm)
dbin2 = 2.   # the uniform bin width > dbin_12trans (nm)
dbin_12trans = 240. # the wavelength switching from dbin1 to dbin2 (nm)

# the frequency to update the actinic flux and optical depth
ini_update_photo_frq = 50
final_update_photo_frq = 5

# ====== Setting up ionchemistry ======
use_ion = False
if use_photo == False and use_ion == True:
    print ('Warning: use_ion = True but use_photo = False')
# photoionization needs to run together with photochemistry

# ====== Setting up parameters for the atmosphere ======
rocky = True # for the surface gravity
nz = 100   # number of vertical layers
use_Kzz = True
use_moldiff = True
use_vz = False
Kzz_prof = 'Pfunc' # Options: 'const','file' or 'Pfunc' (Kzz increased with P^-0.4)
K_max = 1e5        # for Kzz_prof = 'Pfunc'
K_p_lev = 0.1      # for Kzz_prof = 'Pfunc'
vz_prof = 'const'  # Options: 'const' or 'file'
const_Kzz = 1.E10 # (cm^2/s) Only reads when use_Kzz = True and Kzz_prof = 'const'
const_vz = 0 # (cm/s) Only reads when use_vz = True and vz_prof = 'const'


# frequency for updating dz and dzi due to change of mu
update_frq = 50 

# ====== Setting up the boundary conditions ======
# Boundary Conditions:
use_topflux = False
use_botflux = False
max_flux = 1e13  # upper limit for the diffusion-limit fluxes

# ====== Reactions to be switched off  ======
remove_list = [] # in pairs e.g. [1,2]

# == Condensation ======
use_condense = False
use_settling = False
condense_sp = []      
non_gas_sp = []
fix_species = []      # fixed the condensable species after condensation-evapoation EQ has reached  

# ====== steady state check ======
st_factor = 0.5
conv_step = 500

# ====== Setting up numerical parameters for the ODE solver ====== 
ode_solver = 'Ros2' # case sensitive
use_print_prog = True
use_print_delta = False
print_prog_num = 200  # print the progress every x steps 
dttry = 1.E-10
trun_min = 1e2
runtime = 1.E22
dt_min = 1.E-14
dt_max = runtime*1e-5
dt_var_max = 2.
dt_var_min = 0.5
count_min = 120
count_max = int(1E5)

mtol_conv = 1.E-16
pos_cut = 0
nega_cut = -1.
loss_eps = 1e12 # for using BC
yconv_cri = 0.01 # for checking steady-state
slope_cri = 1.e-4
yconv_min = 0.1
flux_cri = 0.1
flux_atol = 1. # the tol for actinc flux (# photons cm-2 s-1 nm-1)

# ====== Setting up numerical parameters for Ros2 ODE solver ====== 
atol = 7.E-3             # Try decreasing this if the solutions are not stable
mtol = 1.E-22
rtol = 0.4              # relative tolerence for adjusting the stepsize 

# ====== Setting up for ouwtput and plotting ======
plot_TP = False
use_live_flux = False
use_plot_end = False
use_plot_evo = False
use_save_movie = False
use_flux_movie = False
plot_height = False
use_PIL = True 
live_plot_frq = 50
save_movie_rate = live_plot_frq
y_time_freq = 1  #  storing data for every 'y_time_freq' step
# output:
output_humanread = False
use_shark = False
save_evolution = False   # save the evolution of chemistry (y_time and t_time) for every save_evo_frq step
save_evo_frq = 50

use_live_plot  = False 
scat_sp = [] 
diff_esc = [] 
atom_list  = ['H', 'O', 'C','N']
network  = 'thermo/NCHO_full_photo_network.txt' 
plot_spec = ['H2','H','H2O','OH','CH4','HCN','N2','NH3'] 
atm_type = 'file'

# ========== PARAMETERS BELOW ARE INSERTED BY OFFLINE_CHEMISTRY AT RUNTIME ===========
