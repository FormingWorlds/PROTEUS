# Import utils-specific modules
from utils.modules_utils import *

from atm_rad_conv.SocRadConv import surf_Planck_nu

# Define Crameri colormaps (+ recursive)
from matplotlib.colors import LinearSegmentedColormap

for name in [ 'acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'buda',
           'cork', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla',
           'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'tofino',
           'tokyo', 'turku', 'vik' ]:
    file = os.path.join(str(pathlib.Path(__file__).parent.absolute())+"/ScientificColourMaps5/", name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name+"_r"] = LinearSegmentedColormap.from_list(name, cm_data[::-1])

vol_colors = {
                "black_1" : "#000000",
                "black_2" : "#323232",
                "black_3" : "#7f7f7f",
                "H2O_1"   : "#8db4cb",
                "H2O_2"   : "#4283A9",
                "H2O_3"   : "#274e65",
                "CO2_1"   : "#811111",
                "CO2_2"   : "#B91919",
                "CO2_3"   : "#ce5e5e",
                "H2_1"    : "#a0d2cb",
                "H2_2"    : "#62B4A9",
                "H2_3"    : "#3a6c65",
                "CH4_1"   : "#eb9194",
                "CH4_2"   : "#E6767A",
                "CH4_3"   : "#b85e61",
                "CO_1"    : "#eab597",
                "CO_2"    : "#DD8452",
                "CO_3"    : "#844f31",
                "N2_1"    : "#c29fb2",
                "N2_2"    : "#9A607F",
                "N2_3"    : "#4d303f",  
                "S_1"     : "#f1ca70",
                "S_2"     : "#EBB434",
                "S_3"     : "#a47d24",    
                "O2_1"    : "#57ccda",
                "O2_2"    : "#2EC0D1",
                "O2_3"    : "#2499a7",
                "He_1"    : "#acbbbf",
                "He_2"    : "#768E95",
                "He_3"    : "#465559"
                }

# Optional command line arguments for running from the terminal
# Usage: $ python plot_atmosphere.py -t 0,718259
parser = argparse.ArgumentParser(description='COUPLER plotting script')
parser.add_argument('-odir', '--output_dir', type=str, help='Full path to output directory');
parser.add_argument('-t', '--times', type=str, help='Comma-separated (no spaces) list of times');
args = parser.parse_args()

# Define output directory for plots
if args.output_dir:
    output_dir = args.output_dir
    print("Output directory:", output_dir)
    
else:
    output_dir = os.getcwd()
    print("Output directory:", output_dir)

# Define which times are plotted
if args.times:
    plot_list = [ int(time) for time in args.times.split(',') ]
    print("Snapshots:", plot_list)
else:
    output_list = su.get_all_output_times(output_dir)

    if len(output_list) <= 8:
        plot_list = output_list
    else:
        plot_list = [ output_list[0], output_list[int(round(len(output_list)*(2./100.)))], output_list[int(round(len(output_list)*(15./100.)))], output_list[int(round(len(output_list)*(22./100.)))], output_list[int(round(len(output_list)*(33./100.)))], output_list[int(round(len(output_list)*(50./100.)))], output_list[int(round(len(output_list)*(66./100.)))], output_list[-1] ]
    print("Snapshots:", plot_list)