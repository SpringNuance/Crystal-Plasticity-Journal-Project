# External libraries
import os
import numpy as np
import optimize_config
import stage1_prepare_data
#import stages_analysis
#import optimization_stages
from modules.SIM_damask2 import *
from stage1_prepare_data import * 
from modules.preprocessing import *
from modules.stoploss import *
from modules.helper import *
from optimizers.GA import *
from optimizers.ANN import *
from optimizers.scaler import * 
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler

def main_trainANN(info, prepared_data, logging):
    logPath = info['logPath']
    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
    curveIndex = info['curveIndex']
    param_info = info['param_info']
    param_info_filtered = info['param_info_filtered']
    paramsUnit = info['paramsUnit']
    material = info['material']
    loadings = info['loadings']
    numberOfHiddenLayers = info['numberOfHiddenLayers']
    hiddenNodesFormula = info['hiddenNodesFormula']
    ANNOptimizer = info['ANNOptimizer']
    L2_regularization = info['L2_regularization']
    learning_rate = info['learning_rate']
    loading_epochs = info['loading_epochs']

    weightsYieldingConstitutive = info['weightsYieldingConstitutive']
    weightsHardeningConstitutive = info['weightsHardeningConstitutive']
    weightsYieldingLinearLoadings = info['weightsYieldingLinearLoadings']
    weightsHardeningLinearLoadings = info['weightsHardeningLinearLoadings']
    weightsHardeningAllLoadings = info['weightsHardeningAllLoadings']

    initial_length = prepared_data['initial_length']
    combined_loadings_interpolateCurves = prepared_data['combined_loadings_interpolateCurves']
    exp_curve = prepared_data['exp_curve']

    iterationResultPath = prepared_data['iterationResultPath']

    initial_loadings_trueCurves = prepared_data['initial_loadings_trueCurves']
    initial_loadings_processCurves = prepared_data['initial_loadings_processCurves']
    initial_loadings_interpolateCurves = prepared_data['initial_loadings_interpolateCurves']
    reverse_initial_loadings_trueCurves = prepared_data['reverse_initial_loadings_trueCurves']
    reverse_initial_loadings_processCurves = prepared_data['reverse_initial_loadings_processCurves']
    reverse_initial_loadings_interpolateCurves = prepared_data['reverse_initial_loadings_interpolateCurves']

    if logging:
        printLog("\n" + 70 * "*" + "\n\n", logPath)
        printLog(f"Step 2: Train the regressors for all loadings with the initial simulations of curve {CPLaw}{curveIndex}\n", logPath)
        printLog(f"ANN model: (parameters) -> (stress values at interpolating strain points)\n", logPath)
        
        stringMessage = "ANN configuration:\n"

        logTable = PrettyTable()
        logTable.field_names = ["ANN configurations", "Choice"]

        logTable.add_row(["Number of hidden layers", numberOfHiddenLayers])
        logTable.add_row(["Hidden layer nodes formula", hiddenNodesFormula])
        logTable.add_row(["ANN Optimizer", ANNOptimizer])
        logTable.add_row(["Learning rate", learning_rate])
        logTable.add_row(["L2 regularization term", L2_regularization])

        for loading in loadings:
            logTable.add_row([f"Epochs of {loading}", loading_epochs[CPLaw][loading]])
        
        stringMessage += logTable.get_string()
        stringMessage += "\n"

        printLog(stringMessage, logPath)

    # The ANN regressors for each loading condition
    regressors = {}
    # The scaler for each loading condition
    scalers = {}
    #The training error:
    trainingErrors = {}

    featureMatrixScaling = np.zeros((2, len(list(param_info_filtered.keys()))))
    powerList = np.zeros(len(list(param_info_filtered.keys())))
    for index, parameter in enumerate(list(param_info_filtered.keys())):
        featureMatrixScaling[:, index] = np.array([param_info_filtered[parameter]["low"], param_info_filtered[parameter]["high"]])
        powerList[index] = param_info_filtered[parameter]["power"]

 
    start = time.time()
    for loading in loadings:
        # All loadings share the same parameters, but different stress values
        paramFeatures = np.array([list(dict(params).values()) for params in list(combined_loadings_interpolateCurves[loading].keys())])

        stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_loadings_interpolateCurves[loading].values())])

        # transforming the data
        scalers[loading] = CustomScaler(featureMatrixScaling, powerList)
        paramFeatures = scalers[loading].transform(paramFeatures)

        # Input and output size of the ANN
        sampleSize = stressLabels.shape[0]
        inputSize = paramFeatures.shape[1]
        outputSize = stressLabels.shape[1]
        
        regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers, sampleSize).to(device)
        trainingErrors[loading] = regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw][loading], L2_regularization)
        if logging:
            printLog(f"------------ {loading} ------------\n", logPath)
            printLog(f"paramFeatures shape is {paramFeatures.shape}\n", logPath)
            printLog(f"stressLabels shape is {stressLabels.shape}\n", logPath)
            printLog(f"Training MSE error: {trainingErrors[loading][-1]}\n", logPath)

    end = time.time()

    if logging:
        printLog(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves[loading])}\n", logPath)
        printLog(f"Finish training ANN for all loadings of curve {CPLaw}{curveIndex}\n", logPath)
        printLog(f"Total training time: {round(end - start, 2)}s\n\n", logPath)

    if not os.path.exists(f"{iterationResultPath}/common/default_curve.npy"):
        tupleParamsStresses = list(reverse_initial_loadings_interpolateCurves.items())
        #print(tupleParamsStresses[0])
        #print(exp_curve["interpolate"])
        #sortedClosestHardening = list(sorted(tupleParamsStresses, key = lambda paramsStresses: lossYieldingAllLinear(exp_curve["interpolate"], paramsStresses[1], loadings, weightsYieldingLinearLoadings, weightsYieldingConstitutive)))
        sortedClosestHardening = list(sorted(tupleParamsStresses, key = lambda paramsStresses: lossHardeningAllLoadings(exp_curve["interpolate"], paramsStresses[1], loadings, weightsHardeningAllLoadings, weightsHardeningConstitutive)))

        # Obtaining the default hardening parameters
        default_params = sortedClosestHardening[0][0]
        default_curve = {}
        default_curve["iteration"] = 0
        default_curve["stage"] = 0
        default_curve["parameters_tuple"] = default_params
        default_curve["parameters_dict"] = dict(default_params)
        default_curve["true"] = reverse_initial_loadings_trueCurves[default_params]
        default_curve["process"] = reverse_initial_loadings_processCurves[default_params]
        default_curve["interpolate"] = reverse_initial_loadings_interpolateCurves[default_params]
        default_curve["true_yielding_loss"] = calculateYieldingLoss(exp_curve["interpolate"], default_curve["interpolate"], loadings)
        default_curve["true_hardening_loss"] = calculateHardeningLoss(exp_curve["interpolate"], default_curve["interpolate"], loadings)
        predicted_curve = {}
        predicted_curve['interpolate'] = {}
        for loading in loadings:
            predictedParams = scalers[loading].transform(np.array(list(default_curve["parameters_dict"].values())).reshape(1, -1))
            #print(predictedParams)
            predicted_curve['interpolate'][loading] = {}
            predicted_curve['interpolate'][loading]['stress'] = regressors[loading].predictOneDimension(predictedParams).flatten()
            #print(predicted_curve['interpolate'][loading]['stress'])
            #time.sleep(60)
        
        default_curve["predicted_yielding_loss"] = calculateYieldingLoss(exp_curve["interpolate"], predicted_curve["interpolate"], loadings)
        default_curve["predicted_hardening_loss"] = calculateHardeningLoss(exp_curve["interpolate"], predicted_curve["interpolate"], loadings)
        print(default_curve["true_yielding_loss"] )
        print(default_curve["predicted_yielding_loss"] )
        print(default_curve["true_hardening_loss"] )
        print(default_curve["predicted_hardening_loss"] )
        np.save(f"{iterationResultPath}/common/default_curve.npy", default_curve)
    else:
        default_curve = np.load(f"{iterationResultPath}/common/default_curve.npy", allow_pickle=True).tolist()
        printLog("The file default_curve.npy exists. Loading the default curves\n", logPath)

    printLog(f"The default parameters for the optimization of curve {CPLaw}{curveIndex}\n", logPath)
    
    printTupleParametersClean(default_curve["parameters_tuple"], param_info, paramsUnit, CPLaw, logPath)
    #time.sleep(60)
    
    trained_models = {
        "regressors": regressors,
        "scalers": scalers,
        "trainingErrors": trainingErrors,
        'default_curve': default_curve,
    }

    #time.sleep(180)

    return trained_models

if __name__ == '__main__':
    info = optimize_config.main_config()
    prepared_data = stage1_prepare_data.main_prepareData(info)
    main_trainANN(info, prepared_data, logging=True)