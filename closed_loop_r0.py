#!/usr/bin/env python

from shesha.config import ParamConfig
from docopt import docopt
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
import matplotlib.pyplot as plt
import os
import numpy as np

savepath='images'

def run_simulation(param_file, r0=None):
    """ Runs a closed-loop simulation given a r0 value, and returns the wfs image as an array.

    ### Parameters
    1. param_file : str
        - path to the parameter file being used for the simulation
    2. r0 : float/int
        - r0 value to be used for simulation

    ### Returns
    - [float]
        - wfs image as an array
    """

    config = ParamConfig(param_file)

    if r0 is not None:
        config.p_atmos.set_r0(float(r0))

    sup = Supervisor(config, silence_tqdm=True)

    sup.loop(sup.config.p_loop.niter)

    wfs = sup.wfs.get_wfs_image(0)

    # save wfs image to ./image/ directory
    #dir = os.path.join(os.path.dirname(__file__), savepath)
    #save_dir = os.path.join(dir, f'wfs{np.round(r0, 2)}.png')
    #plt.imshow(wfs)
    #plt.savefig(save_dir)

    return wfs




          
    
'''
if __name__ == "__main__":
    arguments = docopt(__doc__)

    param_file = arguments["<parameters_filename>"]
    save_file = arguments["<save_filemane>"]

    silence_tqdm = arguments["--silence"]

    config = ParamConfig(param_file)

    # Get parameters from file
    if arguments["--niter"]:
        config.p_loop.set_niter(int(arguments["--niter"]))

    if arguments["--r0"]: 
        config.p_atmos.set_r0(float(arguments["--r0"]))

    supervisor = Supervisor(config, silence_tqdm=silence_tqdm)

    supervisor.loop(supervisor.config.p_loop.niter, compute_tar_psf=compute_tar_psf)
    
    if arguments["--interactive"]:
        from shesha.util.ipython_embed import embed
        from os.path import basename
        embed(basename(__file__), locals())
'''
