# External libraries
import os
import numpy as np
import optimize_config
import stage1_prepare_data
from modules.SIM_damask2 import *
from modules.preprocessing import *
from modules.helper import *
from prettytable import PrettyTable

def main_stagesAnalysis(info, prepared_data):    
    server = info['server']
    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
    initialSims = info['initialSims']
    curveIndex = info['curveIndex']
    projectPath = info['projectPath']
    optimizerName = info['optimizerName']
    param_info = info['param_info']
    param_info_filtered = info['param_info_filtered']
    logPath = info['logPath']
    material = info['material']
    method = info['method']
    searchingSpace = info['searchingSpace']
    roundContinuousDecimals = info['roundContinuousDecimals']
    linearYieldingDev = info['linearYieldingDev']
    linearHardeningDev = info['linearHardeningDev'] 
    nonlinearHardeningDev = info['nonlinearHardeningDev']
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

    iteration_length = prepared_data['iteration_length']

    printLog("\n" + 70 * "*" + "\n\n", logPath)
    printLog(f"Step 3: Assessment of the optimization stages of curve {CPLaw}{curveIndex}\n", logPath)

    allParams = list(param_info_filtered.keys())
    
    yieldingParams = list(filter(lambda param: param_info[param]["type"] == "yielding", allParams))
    linearHardeningParams = list(filter(lambda param: param_info[param]["type"] == "linear_hardening", allParams))
    nonlinearHardeningParams = list(filter(lambda param: param_info[param]["type"] == "nonlinear_hardening", allParams))
    
    printLog(f"The yielding parameters are {yieldingParams}\n", logPath)
    printLog(f"The linear hardening parameters are {linearHardeningParams}\n", logPath)
    printLog(f"The nonlinear hardening parameters are {nonlinearHardeningParams}\n\n", logPath)    
    
    if len(yieldingParams) == 0:
        printLog("There are yielding parameters\n", logPath)
        printLog("1st stage optimization not required\n", logPath)
    else:
        printLog(f"There are {len(yieldingParams)} yielding parameters\n", logPath)
        printLog("1st stage optimization required\n", logPath)
    
    if len(linearHardeningParams) == 0:
        printLog("There are no linear hardening parameters\n", logPath)
        printLog("2nd stage optimization not required\n", logPath)
    else:
        printLog(f"There are {len(linearHardeningParams)} linear hardening parameters\n", logPath)
        printLog("2nd stage optimization required\n", logPath)

    if len(nonlinearHardeningParams) == 0:
        printLog("There are no nonlinear hardening parameters\n", logPath)
        printLog("3rd stage optimization not required\n\n", logPath)
    else:
        printLog(f"There are {len(nonlinearHardeningParams)} small hardening parameters\n", logPath)
        printLog("3rd stage optimization required\n\n", logPath)



    yieldingDevs = {}
    linearHardeningDevs = {}
    nonlinearHardeningDevs = {}
    for loading in loadings:
        if loading.startswith("linear"):
            yieldingDevs[loading] = linearYieldingDev
            linearHardeningDevs[loading] = linearHardeningDev
        else:
            nonlinearHardeningDevs[loading] = nonlinearHardeningDev 
    # ----------------------------------------------------------------------------
    #   Four optimization stage: Optimize the parameters for the curves in parallel 
    # ----------------------------------------------------------------------------
    deviationPercent = [yieldingDevs, linearHardeningDevs, nonlinearHardeningDevs]
    deviationCondition = [insideYieldingDevAllLinear, insideHardeningDevAllLinear, insideHardeningDevAllLoadings]
    optimizeParams = [yieldingParams, linearHardeningParams, nonlinearHardeningParams]
    parameterType = ["yielding", "linear hardening", "nonlinear hardening"]
    optimizeType = ["yielding", "hardening", "hardening"]
    ordinalUpper = ["First", "Second", "Third"]
    ordinalLower = ["first", "second", "third"]
    ordinalNumber = ["1","2","3"]

    stages_data = {
        'deviationPercent':deviationPercent,
        'deviationCondition':deviationCondition,
        'optimizeParams':optimizeParams,
        'parameterType':parameterType,
        'optimizeType':optimizeType,
        'ordinalUpper':ordinalUpper,
        'ordinalLower':ordinalLower,
        'ordinalNumber':ordinalNumber,
    }

    #time.sleep(180)

    return stages_data

if __name__ == '__main__':
    info = optimize_config.main()
    prepared_data = stage1_prepare_data.main_prepareData(info)
    stages_data = main_stagesAnalysis(info, prepared_data)