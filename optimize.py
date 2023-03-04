###########################################################
#                                                         #
#         CRYSTAL PLASTICITY PARAMETER CALIBRATION        #
#   Tools required: DAMASK and Finnish Supercomputer CSC  #
#                                                         #
###########################################################

# External libraries
import os
import numpy as np
import optimize_config
import initial_simulations  
import prepare_data
#import stages_analysis
#import optimization_stages
from modules.SIM_damask2 import *
from prepare_data import * 
from modules.preprocessing import *
from modules.stoploss import *
from modules.helper import *
from optimizers.GA import *
from optimizers.ANN import *
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler

def main_optimize(info):

    def printList(messages):
        for message in messages:
            print(message)

    server = info['server']
    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
    initialSims = info['initialSims']
    curveIndex = info['curveIndex']
    projectPath = info['projectPath']
    optimizerName = info['optimizerName']
    param_info = info['param_info']
    material = info['material']
    method = info['method']
    searchingSpace = info['searchingSpace']
    roundContinuousDecimals = info['roundContinuousDecimals']
    linearYieldingDev = info['linearYieldingDev']
    firstLinearHardeningDev = info['firstLinearHardeningDev'] 
    secondLinearHardeningDev = info['secondLinearHardeningDev']
    firstNonlinearHardeningDev = info['firstNonlinearHardeningDev']
    secondNonlinearHardeningDev = info['secondNonlinearHardeningDev']
    loadings = info['loadings']
    exampleLoading = info['exampleLoading']
    yieldingPoints = info['yieldingPoints']
    weightsYielding = info['weightsYielding']
    weightsHardening = info['weightsHardening']
    weightsLoading = info['weightsLoading']
    paramsFormatted = info['paramsFormatted']
    paramsUnit = info['paramsUnit']
    numberOfHiddenLayers = info['numberOfHiddenLayers']
    hiddenNodesFormula = info['hiddenNodesFormula']
    ANNOptimizer = info['ANNOptimizer']
    L2_regularization = info['L2_regularization']
    learning_rate = info['learning_rate']
    loading_epochs = info['loading_epochs']
    
    # -------------------------------------------------------------------
    #   Step 1: Running initial simulations
    # -------------------------------------------------------------------

    initial_simulations.main_initial_sims(info)

    # -------------------------------------------------------------------
    #   Step 2: Extracting the experimental and simulated data
    # -------------------------------------------------------------------

    prepared_data = prepare_data.main_prepare(info)
    
    # -------------------------------------------------------------------
    #   Step 4: Analyzing the optimization stages
    # -------------------------------------------------------------------

    #analyzed_stages= stages_analysis.main_stagesAnalysis(info)

    # -------------------------------------------------------------------
    #   Step 5: Optimize the parameters for the curves
    # -------------------------------------------------------------------        

    
    # Outside the for-loop of 4 optimization stages

    messages = [f"All four optimization stages have successfully completed for curve {CPLaw}{curveIndex}\n"]

    ########
    stringMessage = "The final optimized set of parameters is:\n"
    
    logTable = PrettyTable()
    logTable.field_names = ["Parameter", "Value"]

    stage4_curves = np.load(f"{iterationPath}/stage4_curves.npy", allow_pickle=True).tolist()

    for param in stage4_curves['parameters_dict']:
        paramValue = stage4_curves['parameters_dict'][param]
        exponent = param_info[param]['exponent'] if param_info[param]['exponent'] != "e0" else ""
        unit = paramsUnit[CPLaw][param]
        paramString = f"{paramValue}"
        if exponent != "":
            paramString += exponent
        if unit != "":
            paramString += f" {unit}"
        logTable.add_row([param, paramString])

    stringMessage += logTable.get_string()
    stringMessage += "\n"
    messages.append(stringMessage)  
    ########

    messages.append(f"Optimization for curve {CPLaw}{curveIndex} has finished\n")

    printList(messages, curveIndex)

    # ------------------------------
    #   Finalizing the optimization 
    # ------------------------------

    print("\n" + 70 * "=" + "\n")
    print("Fitting parameter optimization for all target curves completed\n")
    print("Congratulations! Thank you for using the Crystal Plasticity Software\n")

if __name__ == '__main__':

    # -------------------------------------------------------------------
    #   Step 0: Defining the initial simulation configurations
    # -------------------------------------------------------------------

    info = optimize_config.main_config()
    main_optimize(info)
