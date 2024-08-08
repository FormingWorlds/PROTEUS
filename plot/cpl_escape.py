#!/usr/bin/env python3

from utils.modules_ext import *
from utils.plot import *

log = logging.getLogger("PROTEUS")

def plot_escape(output_dir, escape_model, plot_format="pdf", t0=100.0) :

    log.info("Plot escape")

    hf_all = pd.read_csv(output_dir+"/runtime_helpfile.csv", sep=r"\s+")

    time = np.array(hf_all["Time"])
    if len(time) < 3:
        log.warning("Cannot make plot with less than 3 samples")
        return

    # make plot
    lw = 1.2
    al = 0.5
    scale = 1.1
    fig,ax = plt.subplots(1,1, figsize=(7*scale,4*scale))

    if escape_model == 0 :
        escape_model_label = 'No escape'
    elif escape_model == 1 :
        escape_model_label = 'Energy-limited escape (Zephyrus)'
    elif escape_model == 2 :
        escape_model_label = 'Dummy escape'
    
    y = hf_all['esc_rate_total']
    l = ax.plot(time, y, lw=lw, ls='solid', label=f'{escape_model_label}')


    # decorate 
    ax.set_ylabel(r'Mass loss rate [kg $s^{-1}$]')
    ax.set_yscale("log")
    ax.set_xlabel("Time [yr]")
    ax.set_xscale("log")
    ax.set_xlim(left=t0, right=np.amax(time))
    ax.grid(alpha=0.2)
    ax.legend()

    plt.close()
    plt.ioff()

    fpath = os.path.join(output_dir, "plot_escape.%s"%plot_format)
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
    COUPLER_options = ReadInitFile( cfg )

    # Set directories dictionary
    dirs = SetDirectories(COUPLER_options)

    plot_escape(output_dir=dirs["output"], escape_model=COUPLER_options['escape_model'], plot_format=COUPLER_options["plot_format"])

#====================================================================

if __name__ == "__main__":

    main()
