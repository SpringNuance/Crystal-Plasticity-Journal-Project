#############################################
#                                           #
#        Running initial simulations        #
#                                           #
#############################################

import os
import numpy as np
import optimize_config
from modules.SIM_damask2 import *
from modules.preprocessing import *
from modules.helper import *

def main_initial_sims(info):

    ############################################
    # Generating universal initial simulations #
    ############################################
    # External libraries

    print("\n" + 70 * "*" + "\n\n")
    print(f"Step 0: Running initial simulations\n")
    
    method = info["method"]
    material = info['material']
    CPLaw = info['CPLaw']
    loadings = info['loadings']
    exampleLoading = info['exampleLoading']

    loadings_not_having_initial_sims = []
    for loading in loadings:
        if not os.path.exists(f"results/{material}/{CPLaw}/universal/{loading}/initial_processCurves.npy"):
            loadings_not_having_initial_sims.append(loadings)
    if len(loadings_not_having_initial_sims) != 0:
        print("Loadings not having initial simulations:")
        for loading in loadings:
            print(loading)
        print("Starting initial simulations\n")
        info_new_loadings = copy.deepcopy(info)
        info_new_loadings["loadings"] = loadings_not_having_initial_sims
        simUniversal = SIM(info_new_loadings)
        manualParams = np.load(f"manualParams/{material}/{CPLaw}/initial_params.npy", allow_pickle=True)
        if method == "manual":     
            tupleParams = manualParams[0:25] # <-- Run the parameters in small batches
            simUniversal.run_initial_simulations(tupleParams)
        elif method == "auto":
            simUniversal.run_initial_simulations()
        initial_processCurves = np.load(f'results/{material}/{CPLaw}/universal/initial_processCurves.npy', allow_pickle=True).tolist()
        initial_length = len(initial_processCurves)
        print(f"Initial simulations have finished. {initial_length} simulations completed\n")
    else:
        initial_processCurves = np.load(f'results/{material}/{CPLaw}/universal/{exampleLoading}/initial_processCurves.npy', allow_pickle=True).tolist()
        initial_length = len(initial_processCurves)
        print(f"Initial simulations have finished. {initial_length} simulations completed\n")

if __name__ == '__main__':
    info = optimize_config.main_config()
    main_initial_sims(info)
