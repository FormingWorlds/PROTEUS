#!/usr/bin/env python3

"""
Code for running offline chemistry based on PROTEUS output
"""

# Import libraries
import numpy as np
import pickle as pkl
import os, glob, shutil, subprocess, time, pathlib
from datetime import datetime
import numpy.random as nrand
import logging

# Import PROTEUS stuff
from utils.coupler import *

# ------------------------------------------------------------------------------------------------

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)

def run_once(year:int, now:int, first_run:bool, COUPLER_options:dict) -> bool:
    """Run VULCAN once for a given PROTEUS output year
    
    Runs VULCAN in a screen instance, so that lots processes may still be
    managed individually. Does not copy final output to PROTEUS folder, but 
    leaves it in VULCAN/output for the master thread to copy later on.

    Parameters
    ----------
        year : int
            Planet age corresponding to a data dump from PROTEUS [yr]
        now : int
            Time at which this run was started (format: HHMMSS)
        first_run : bool
            Is this the first time that VULCAN is run for this configuration?
        COUPLER_options : dict
            PROTEUS options dictionary read from cfg file
    Returns
    ----------
        success : bool
            Did everything run okay?
    """

    success = True

    mixing_ratio_floor = 1e-10

    # Copy template config file
    shutil.copyfile(dirs["utils"]+"/vulcan_offline_template.py",dirs["vulcan"]+"vulcan_cfg.py")

    # Make results folder for this run
    this_results = dirs["output"]+"offchem/%d/"%year
    os.makedirs(this_results)

    # Read atm object
    with open(dirs["output"]+"%d_atm.pkl"%year,'rb') as atm_handle:
        atm = pkl.load(atm_handle)
        p_bot   = atm.pl[-1]        # Pressure
        p_top   = atm.pl[0]         # Pressure
        v_mx    = atm.vol_list      # Mixing ratios
        
        volatile_species = v_mx.keys()

    # Write specifics to config file
    with open(dirs["vulcan"]+"vulcan_cfg.py",'a') as vcf:
        # Output file to use
        vcf.write("out_name = '%d_offchem_%d.vul' \n" % (now,year))

        # Pressure grid limits
        vcf.write("P_b = %1.5e \n" % float(p_bot * 10.0))  # pressure at the bottom (dyne/cm^2)
        vcf.write("P_t = %1.5e \n" % float(p_top * 10.0))  # pressure at the top (dyne/cm^2)

        # Bottom boundary condition
        fix_bb_mr = "{"
        for v in volatile_species:
            this_mx = max(float(v_mx[v]),mixing_ratio_floor)
            if (this_mx > 1e-1):
                fix_bb_mr += " '%s' : %1.5e ," % (v,this_mx)
        fix_bb_mr = fix_bb_mr[:-1]+" }"
        vcf.write("use_fix_sp_bot = %s \n" % str(fix_bb_mr))

        # Initial constant mixing ratios
        const_mix = "{"
        for v in volatile_species:
            this_mx = max(float(v_mx[v]),mixing_ratio_floor)      # Mixing ratio floor to prevent issues with VULCAN
            const_mix += " '%s' : %1.5e ," % (v,this_mx)
        const_mix = const_mix[:-1]+" }"
        vcf.write("const_mix = %s \n" % str(const_mix))

        # PT profile
        vcf.write("atm_file = 'output/%d_offchem_%d_PT.txt' \n"%(now,year))

        # Background componen of atmosphere
        # TODO - CODE BELOW IS TEMPORARY
        vcf.write("atm_base = 'H2' \n")

        # Stellar spectrum
        # TODO - CODE BELOW IS TEMPORARY
        vcf.write("sflux_file = 'atm/stellar_flux/Gueymard_solar.txt' \n")

        # Stellar radius
        # TODO - CODE BELOW IS TEMPORARY
        vcf.write("r_star = %1.6e \n" %         float(COUPLER_options["star_radius_modern"]))

        # Other planet parameters
        vcf.write("Rp = %1.6e \n" %             float(COUPLER_options["radius"]*100.0))
        vcf.write("gs = %1.6e \n" %             float(COUPLER_options["gravity"]*100.0))
        vcf.write("sl_angle = %1.6e \n" %       float(COUPLER_options["zenith_angle"]*np.pi/180.0))
        vcf.write("orbit_radius = %1.6e \n" %   float(COUPLER_options["mean_distance"]))

    
    # Copy this config file to results dir for posterity
    shutil.copyfile(dirs["vulcan"]+"vulcan_cfg.py",this_results+"vulcan_cfg.py")

    # Write PT profile
    vul_PT = np.array(
        [np.array(atm.pl)  [::-1] * 10.0,
         np.array(atm.tmpl)[::-1]
        ]
    ).T
    header = "#(dyne/cm2)\t (K) \n Pressure\t Temp"
    np.savetxt(dirs["vulcan"]+"output/%d_offchem_%d_PT.txt"%(now,year),vul_PT,  delimiter="\t",header=header,comments='',fmt="%1.5e")
    shutil.copyfile(dirs["vulcan"]+"output/%d_offchem_%d_PT.txt"%(now,year),    dirs["output"]+"offchem/%d/PT.txt"%year)

    # Run vulcan
    # Note that the screen name ends with an 'x' so that accurate matching can be done with grep
    os.chdir(dirs["vulcan"])
    screen_name = "%d_offchem_%dx" % (now,year)
    logfile = dirs["output"]+"offchem/%d/vulcan.log"%year
    vulcan_flag = " " if first_run else "-n"
    vulcan_run = 'screen -S %s -L -Logfile %s -dm bash -c "python3 vulcan.py %s"' % (screen_name,logfile,vulcan_flag)
    subprocess.run([vulcan_run], shell=True, check=True)
    os.chdir(dirs["coupler"])

    # Kill these screens with the command below (from: https://stackoverflow.com/a/8987063)
    # Run: `pkill -f [datetime]_offchem_`

    return success


# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    print("Started main process")

    # Parameters
    samples =   30                  # How many samples to use from /output/
    cfgfile =   "init_coupler.cfg"  # Config file used for PROTEUS

    # Read in PROTEUS config file
    COUPLER_options, time_dict = ReadInitFile( cfgfile )

    # Set directories
    dirs = SetDirectories(COUPLER_options)
    offchem_dir = dirs["output"]+"offchem/"

    # Copy cfg file for posterity
    shutil.copyfile(cfgfile, offchem_dir+cfgfile)

    # Delete old files
    if os.path.exists(offchem_dir):
        shutil.rmtree(offchem_dir)
    pathlib.Path(offchem_dir).mkdir()
    if os.path.isdir(offchem_dir):
        if os.listdir(offchem_dir):
            raise Exception("Could not clear directory for new run")
    else:
        raise Exception("Could not make directory '%s'" % offchem_dir)

    # Set up logging
    logfile = offchem_dir+"parent.log"
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler()
        ]
    )

    # Log info to user
    logging.info("Date: " +str(datetime.now().strftime("%Y-%m-%d")))

    now = int(str(datetime.now().strftime("%H%M%S")))
    logging.info("Time: %d"%now)

    logging.info(" ")
    logging.info("This program will generate several screen sessions")
    logging.info("To kill all of them, run `pkill -f %d_offchem_`" % now)
    logging.info("Take care to avoid orphaned instances using `screen -ls`")
    logging.info(" ")

    # Find out which years we have both SPIDER and AEOLUS data for
    evolution_json = glob.glob(dirs["output"]+"*.json")
    json_years = np.array([int(f.split("/")[-1].split(".")[0]) for f in evolution_json])
    json_years = np.sort(json_years)
    logging.info("Number of json files: %d"%len(json_years))

    evolution_pkl = glob.glob(dirs["output"]+"*_atm.pkl")
    pkl_years = np.array([int(f.split("/")[-1].split("_atm.")[0]) for f in evolution_pkl])
    pkl_years = np.sort(pkl_years)
    logging.info("Number of pkl files: %d"%len(pkl_years))

    years_all = []
    for y in json_years:
        if y in pkl_years: 
            years_all.append(y)

    # Select samples...
    if samples < 1:
        raise Exception("Too few samples requested! (Less than zero)") 
    if samples > len(years_all):
        raise Exception("Too many samples requested! (Duplicates expected)") 
    if samples > 60:
        raise Exception("Too many samples requested! (Child process count is capped)") 
    
    logging.info("Choosing sample years...")
    years = [ ]
    yfirst = years_all[0]
    ylast = years_all[-1]
    if samples >= 1:
        years.append(yfirst) # Include first run
    if samples >= 2:
        years.append(ylast) # Include last run

    # Draw other runs randomly from a distribution. This is set-up to sample
    # the end of the run more densely than the start, because the evolution at 
    # the start isn't so interesting compared to the end.
    sample_itermax = 100*samples
    sample_iter = 0
    while (len(years) < samples): 

        sample_iter += 1
        if sample_iter > sample_itermax:
            raise Exception("Maximum iterations reached in selecting sample years!")

        sample = np.abs(nrand.laplace(loc=ylast,scale=ylast*100.0))
        if (sample <= yfirst) or (sample >= ylast):
            continue  # Try again

        inear = find_nearest_idx(years_all,sample)
        ynear = years_all[inear]
        if ynear in years:
            continue  # Try again
        
        years.append(ynear)

    years = sorted(set(years))  # Ensure that there are no duplicates and sort ascending
    logging.info("Years selected: " +str(years))

    # Start processes
    logging.info(" ")
    logging.info("Starting %d processses..." % samples)
    for i,y in enumerate(years):
        if (i == 0):
            time.sleep(3.0)   # Wait 5 seconds for first process to allow for cancelling
        elif (i == 1):
            time.sleep(60.0)  # Wait for chem_funs.py to finish making the network py file
        else:
           time.sleep(45.0)  # Wait before starting next process to allow VULCAN to initialise

        logging.info("\t %03d idx : %08d yrs" % (i,y))
        this_success = run_once(y, now,bool(i==0),COUPLER_options)
        if not this_success:  logging.info("\t    WARNING: could not dispatch this ^ VULCAN instance!")
        

    logging.info("Completed VULCAN processes...")
    running = [1]
    while len(running) > 0:
        running = []
        screen_ls = subprocess.run(['screen','-ls'], stdout=subprocess.PIPE)
        screen_out = str(screen_ls.stdout.decode("utf-8"))
        for y in years:
            if str("offchem_%dx"%y) in screen_out:
                running.append(y)
        logging.info("\t %03d / %03d" % (len(years)-len(running),len(years)))

        # Copy output files
        for y in years:
            to_copy = dirs["vulcan"]+"/output/%d_offchem_%d.vul" % (now,y)
            if os.path.exists(to_copy):
                shutil.copyfile(to_copy, dirs["output"]+"/offchem/%d/output.vul"%y)

        time.sleep(20.0)  # Wait a while before checking again

    logging.info("All processes finished at: "+str(datetime.now().strftime("%H%M%S")))

    logging.info("Done!")
