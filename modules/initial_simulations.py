#############################################
#                                           #
#        Running initial simulations        #
#                                           #
#############################################



import os
import numpy as np
from threading import Lock
from modules.SIM import *
from modules.preprocessing import *
from modules.stoploss import *
from modules.helper import *
from optimizers.GA import *
from optimizers.ANN import *
from prettytable import PrettyTable
from optimize_config import *
from sklearn.preprocessing import StandardScaler

def runInitialSimulations():
    # Loading the parameter information
    #                                                                                    numOfColumns, startingColumn, spacing, nrows
    getParamRanges(material, CPLaw, curveIndices, searchingSpace, searchingType, roundContinuousDecimals, 3, 9, 1)
    general_param_info = loadGeneralParam(material, CPLaw)

    info = {
        'param_info': general_param_info,
        'server': server,
        'loadings': loadings,
        'CPLaw': CPLaw,
        'initialSims': initialSims,
        'projectPath': projectPath,
        'optimizerName': optimizerName,
        'material': material,
        'method': method,
        'searchingSpace': searchingSpace,
        'roundContinuousDecimals': roundContinuousDecimals,
        'loadings': loadings
    }

    ############################################
    # Generating universal initial simulations #
    ############################################

    print("\n" + 70 * "*" + "\n\n")
    print(f"Step 0: Running initial simulations\n\n")

    simUniversal = SIM(info)
    if method == "manual":
        print("Starting initial simulations\n")
        manualParams = np.load(f"manualParams/{material}/{CPLaw}/initial_params.npy", allow_pickle=True)
        tupleParams = manualParams[0:25] # <-- Run the parameters in small batches
        simUniversal.run_initial_simulations(tupleParams)
        print(f"Done. {len(tupleParams)} simulations completed.")
    elif method == "auto":
        if not os.path.exists(f"results/{material}/{CPLaw}/universal/initial_processCurves.npy"):
            print("Starting initial simulations\n")
            simUniversal.run_initial_simulations()
        initial_processCurves = np.load(f'results/{material}/{CPLaw}/universal/initial_processCurves.npy', allow_pickle=True).tolist()
        print(f"Initial simulations finished. {len(initial_processCurves['linear_uniaxial_RD'])} simulations completed\n")

