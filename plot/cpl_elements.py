#!/usr/bin/env python3

from proteus.utils.modules_ext import *
from proteus.utils.plot import *

log = logging.getLogger("PROTEUS")

def plot_elements( output_dir, plot_format="pdf", t0=100.0):

    log.info("Plot elements")

    hf_all = pd.read_csv(output_dir+"/runtime_helpfile.csv", sep=r"\s+")

    time = np.array(hf_all["Time"] )
    if len(time) < 3:
        log.warning("Cannot make plot with less than 3 samples")
        return

    # make plot
    lw = 1.2
    al = 0.5
    scale = 1.1
    fig,ax = plt.subplots(1,1, figsize=(7*scale,4*scale))


    total = np.zeros(len(time))
    for e in element_list:
        y = hf_all[e+"_kg_total"]
        l = ax.plot(time, y, lw=lw, ls='solid', label="Total "+e)[0]

        total += y

        # c = l.get_color()
        # ax.plot(time, hf_all[e+"_kg_liquid"], lw=lw, color=c, alpha=al, ls='dashed')  # in magma ocean
        # ax.plot(time, hf_all[e+"_kg_atm"],    lw=lw, color=c, alpha=al, ls='dotted')  # in atmosphere
    
    ax.plot(time, total,           lw=lw, ls='solid',  label='Total',  c='k')
    ax.plot(time, hf_all["M_atm"], lw=lw, ls='dotted', label='Atmos.', c='k')

    # decorate 
    ax.set_ylabel("Inventory [kg]")
    ax.set_yscale("log")
    ax.set_xlabel("Time [yr]")
    ax.set_xscale("log")
    ax.set_xlim(left=t0, right=np.amax(time))
    ax.grid(alpha=0.2)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_elements.%s"%plot_format)
    fig.savefig(fpath, dpi=200, bbox_inches='tight')

#====================================================================
def main():

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 

    # Read in COUPLER input file
    log.info("Read cfg file")
    from utils.coupler import ReadInitFile, SetDirectories
    OPTIONS = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(OPTIONS)

    plot_elements( output_dir=dirs["output"], plot_format=OPTIONS["plot_format"] )

#====================================================================

if __name__ == "__main__":

    main()
