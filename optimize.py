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
import stage0_initial_simulations  
import stage1_prepare_data
import stage2_train_ANN
import stage3_stages_analysis
import stage4_SOO
import stage4_MOO
from modules.SIM_damask2 import *
from stage1_prepare_data import * 
from modules.preprocessing import *
from modules.stoploss import *
from modules.helper import *
from optimizers.GA import *
from optimizers.ANN import *
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler

def main_optimize(info):

    server = info['server']
    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
    initialSimsSpacing = info['initialSimsSpacing']
    initialSims = info['initialSims']
    curveIndex = info['curveIndex']
    projectPath = info['projectPath']
    optimizeStrategy = info['optimizeStrategy']
    optimizerName = info['optimizerName']
    param_info = info['param_info']
    param_info_GA = info['param_info_GA']
    param_info_BO = info['param_info_BO']
    param_info_PSO = info['param_info_PSO']
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
    weightsYieldingConstitutive = info['weightsYieldingConstitutive']
    weightsHardeningConstitutive = info['weightsHardeningConstitutive']
    weightsYieldingLinearLoadings = info['weightsYieldingLinearLoadings']
    weightsHardeningLinearLoadings = info['weightsHardeningLinearLoadings']
    weightsHardeningAllLoadings = info['weightsHardeningAllLoadings']
    paramsFormatted = info['paramsFormatted']
    paramsUnit = info['paramsUnit']
    numberOfHiddenLayers = info['numberOfHiddenLayers']
    hiddenNodesFormula = info['hiddenNodesFormula']
    ANNOptimizer = info['ANNOptimizer']
    L2_regularization = info['L2_regularization']
    learning_rate = info['learning_rate']
    loading_epochs = info['loading_epochs']
    
    # -------------------------------------------------------------------
    #   Step 0: Running initial simulations
    # -------------------------------------------------------------------

    stage0_initial_simulations.main_initialSims(info)

    # -------------------------------------------------------------------
    #   Step 1: Extracting the experimental and simulated data
    # -------------------------------------------------------------------

    prepared_data = stage1_prepare_data.main_prepareData(info)


    initial_length = prepared_data['initial_length']
    iteration_length = prepared_data['iteration_length']
    exp_curve = prepared_data['exp_curve']
    initialResultPath = prepared_data['initialResultPath']
    iterationResultPath = prepared_data['iterationResultPath']
    stage_CurvesList = prepared_data['stage_CurvesList']

    initial_loadings_trueCurves = prepared_data['initial_loadings_trueCurves']
    initial_loadings_processCurves = prepared_data['initial_loadings_processCurves']
    initial_loadings_interpolateCurves = prepared_data['initial_loadings_interpolateCurves']
    reverse_initial_loadings_trueCurves = prepared_data['reverse_initial_loadings_trueCurves']
    reverse_initial_loadings_processCurves = prepared_data['reverse_initial_loadings_processCurves']
    reverse_initial_loadings_interpolateCurves = prepared_data['reverse_initial_loadings_interpolateCurves']
    iteration_loadings_trueCurves = prepared_data['iteration_loadings_trueCurves']
    iteration_loadings_processCurves = prepared_data['iteration_loadings_processCurves']
    iteration_loadings_interpolateCurves = prepared_data['iteration_loadings_interpolateCurves']
    reverse_iteration_loadings_trueCurves = prepared_data['reverse_iteration_loadings_trueCurves']
    reverse_iteration_loadings_processCurves = prepared_data['reverse_iteration_loadings_processCurves']
    reverse_iteration_loadings_interpolateCurves = prepared_data['reverse_iteration_loadings_interpolateCurves']
    combined_loadings_trueCurves = prepared_data['combined_loadings_trueCurves']
    combined_loadings_processCurves = prepared_data['combined_loadings_processCurves']
    combined_loadings_interpolateCurves = prepared_data['combined_loadings_interpolateCurves']
    reverse_combined_loadings_trueCurves = prepared_data['reverse_combined_loadings_trueCurves']
    reverse_combined_loadings_processCurves = prepared_data['reverse_combined_loadings_processCurves']
    reverse_combined_loadings_interpolateCurves = prepared_data['reverse_combined_loadings_interpolateCurves']
    
    # -------------------------------------------------------------------
    #   Step 2: Training the ANN models
    # -------------------------------------------------------------------

    trained_models = stage2_train_ANN.main_trainANN(info, prepared_data, logging=True)

    regressors = trained_models["regressors"]
    scalers = trained_models["scalers"]
    trainingErrors = trained_models["trainingErrors"]
    default_curve = trained_models['default_curve']

    # -------------------------------------------------------------------
    #   Step 3: Analyzing the optimization stages
    # -------------------------------------------------------------------

    stages_data = stage3_stages_analysis.main_stagesAnalysis(info, prepared_data)

    deviationPercent_stages = stages_data['deviationPercent_stages']
    stopFunction_stages = stages_data['stopFunction_stages']
    lossFunction_stages = stages_data['lossFunction_stages']
    optimizeParams_stages = stages_data['optimizeParams_stages']
    weightsLoadings_stages = stages_data['weightsLoadings_stages']
    weightsConstitutive_stages = stages_data['weightsConstitutive_stages']
    parameterType_stages = stages_data['parameterType_stages']
    optimizeType_stages = stages_data['optimizeType_stages']
    ordinalUpper_stages = stages_data['ordinalUpper_stages']
    ordinalLower_stages = stages_data['ordinalLower_stages']
    ordinalNumber_stages = stages_data['ordinalNumber_stages']

    # -------------------------------------------------------------------
    #   Step 4: Optimize the parameters for the curves
    # -------------------------------------------------------------------        
    
    if optimizeStrategy == "SOO":
        stage4_SOO.main_SOO(info, prepared_data, stages_data, trained_models)
    elif optimizeStrategy == "MOO":
        stage4_MOO.main_MOO(info, prepared_data, stages_data, trained_models)
    # Outside the for-loop of 4 optimization stages

    printLog(f"All four optimization stages have successfully completed for curve {CPLaw}{curveIndex}\n", logPath)
    printLog("The final optimized set of parameters is:\n")
    stage3_curves = np.load(f"{iterationResultPath}/stage3_curves.npy", allow_pickle=True).tolist()
    printDictParametersClean(stage3_curves['parameters_dict'], param_info, paramsUnit, CPLaw, logPath)
    printLog(f"Optimization for curve {CPLaw}{curveIndex} has finished\n", logPath)

    # ------------------------------
    #   Finalizing the optimization 
    # ------------------------------

    printLog("\n" + 70 * "=" + "\n", logPath)
    printLog(f"Parameter optimization for the target curve {CPLaw}{curveIndex} completed\n", logPath)
    printLog("Congratulations! Thank you for using the Crystal Plasticity Software\n", logPath)

if __name__ == '__main__':
    info = optimize_config.main_config()
    main_optimize(info)
