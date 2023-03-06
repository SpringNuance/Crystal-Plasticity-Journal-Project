# External libraries
import os
import numpy as np
import optimize_config
import initial_simulations  
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
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler

def main_trainANN(info, prepared_data):

    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
    curveIndex = info['curveIndex']
    param_info = info['param_info']
    material = info['material']
    loadings = info['loadings']
    numberOfHiddenLayers = info['numberOfHiddenLayers']
    hiddenNodesFormula = info['hiddenNodesFormula']
    ANNOptimizer = info['ANNOptimizer']
    L2_regularization = info['L2_regularization']
    learning_rate = info['learning_rate']
    loading_epochs = info['loading_epochs']


    initial_length = prepared_data['initial_length']
    combined_loadings_interpolateCurves = prepared_data['combined_loadings_interpolateCurves']
    

    print(70 * "*" + "\n")
    print(f"Step 2: Train the regressors for all loadings with the initial simulations of curve {CPLaw}{curveIndex}\n")
    print(f"ANN model: (parameters) -> (stress values at interpolating strain points)\n")
    
    stringMessage = "ANN configuration:\n"

    logTable = PrettyTable()
    logTable.field_names = ["ANN configurations", "Choice"]

    logTable.add_row(["Number of hidden layers", numberOfHiddenLayers])
    logTable.add_row(["Hidden layer nodes formula", hiddenNodesFormula])
    logTable.add_row(["ANN Optimizer", ANNOptimizer])
    logTable.add_row(["Learning rate", learning_rate])
    logTable.add_row(["L2 regularization term", L2_regularization])

    # The ANN regressors for each loading condition
    regressors = {}
    # The regularization scaler for each loading condition
    scalers = {}
    
    for loading in loadings:
        logTable.add_row([f"Epochs of {loading}", loading_epochs[CPLaw][loading]])
    
    stringMessage += logTable.get_string()
    stringMessage += "\n"

    print(stringMessage)

    
    
    start = time.time()
    for loading in loadings:
        # All loadings share the same parameters, but different stress values
        paramFeatures = np.array([list(dict(params).values()) for params in list(combined_loadings_interpolateCurves[loading].keys())])
        stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_loadings_interpolateCurves[loading].values())])

        # Normalizing the data
        scalers[loading] = StandardScaler().fit(paramFeatures[0:initial_length])
        paramFeatures = scalers[loading].transform(paramFeatures)

        # Input and output size of the ANN
        sampleSize = stressLabels.shape[0]
        inputSize = paramFeatures.shape[1]
        outputSize = stressLabels.shape[1]

        print(f"({loading}) paramFeatures shape is {paramFeatures.shape}\n")
        print(f"({loading}) stressLabels shape is {stressLabels.shape}\n")

        regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers, sampleSize).to(device)
        trainingError = regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw][loading], L2_regularization)

        print(f"Finish training ANN for loading {loading}\n")
        print(f"Training MSE error: {trainingError[-1]}\n\n")

    end = time.time()

    print(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves[loading])}\n")
    print(f"Finish training ANN for all loadings of curve {CPLaw}{curveIndex}\n")
    print(f"Total training time: {round(end - start, 2)}s\n\n")

    trained_ANN = {
        "regressors": regressors,
        "scalers": scalers
    }

    time.sleep(180)

    return trained_ANN

if __name__ == '__main__':
    info = optimize_config.main()
    main_trainANN(info, prepared_data)