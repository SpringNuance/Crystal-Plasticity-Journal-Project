#############################################
#                                           #
#        Running initial simulations        #
#                                           #
#############################################

import os
import numpy as nplibraries
import optimize_config
from modules.SIM_damask2 import *
from modules.preprocessing import *
from modules.helper import *

def main_initialSims(info):

    ############################################
    # Generating universal initial simulations #
    ############################################
    # External libraries
    logPath = info['logPath']
    printLog("\n" + 70 * "*" + "\n\n", logPath)
    printLog(f"Step 0: Running initial simulations\n", logPath)
    
    method = info["method"]
    material = info['material']
    CPLaw = info['CPLaw']
    loadings = info['loadings']
    initialSims = info['initialSims']
    exampleLoading = info['exampleLoading']
    logPath = info['logPath']

    loadings_not_having_initial_sims = []
    for loading in loadings:
        if not os.path.exists(f"results/{material}/{CPLaw}/universal/{loading}/initial_processCurves.npy"):
            loadings_not_having_initial_sims.append(loading)
    if len(loadings_not_having_initial_sims) != 0:
        printLog("Loadings not having initial simulations:", logPath)
        for loading in loadings:
            printLog(loading, logPath)
        printLog("Starting initial simulations\n", logPath)
        info_new_loadings = copy.deepcopy(info)
        info_new_loadings["loadings"] = loadings_not_having_initial_sims
        simUniversal = SIM(info_new_loadings)
        if not os.path.exists(f"manualParams/{material}/{CPLaw}/initial_params.npy"):
            manualParams = simUniversal.latin_hypercube_sampling(initialSims)
            np.save(f"manualParams/{material}/{CPLaw}/initial_params.npy", manualParams)
            print("\nInitial parameters generated")
            time.sleep(30)
        else: 
            manualParams = np.load(f"manualParams/{material}/{CPLaw}/initial_params.npy", allow_pickle=True)
        
        dictParamsList = manualParams[0:25] # <-- Run the parameters in small batches
        simUniversal.run_initial_simulations(dictParamsList)

        initial_processCurves = np.load(f'results/{material}/{CPLaw}/universal/{exampleLoading}/initial_processCurves.npy', allow_pickle=True).tolist()
        initial_length = len(initial_processCurves)
        printLog(f"Initial simulations have finished. {initial_length} simulations completed\n", logPath)
    else:
        initial_processCurves = np.load(f'results/{material}/{CPLaw}/universal/{exampleLoading}/initial_processCurves.npy', allow_pickle=True).tolist()
        initial_length = len(initial_processCurves)
        printLog(f"Initial simulations have finished. {initial_length} simulations completed\n", logPath)
    
    
if __name__ == '__main__':
    info = optimize_config.main_config()
    main_initialSims(info)
