import numpy as np
from optimizers.ANN import *

def retrainingANN(optimizeType, loadings, regressors, scalers, combined_loadings_interpolateCurves, info):
    CPLaw = info['CPLaw']
    loadings = info['loadings']
    numberOfHiddenLayers = info['numberOfHiddenLayers']
    hiddenNodesFormula = info['hiddenNodesFormula']
    ANNOptimizer = info['ANNOptimizer']
    L2_regularization = info['L2_regularization']
    learning_rate = info['learning_rate']
    loading_epochs = info['loading_epochs']

    # In order to prevent repeated params, retraining ANN with slightly different configuration
    if optimizeType == "yielding":
        # All loadings share the same parameters, but different stress values
        for loading in loadings:
            if loading.startswith("linear"):
                paramFeatures = np.array([list(dict(params).values()) for params in list(combined_loadings_interpolateCurves[loading].keys())])

                # Normalizing the data
                paramFeatures = scalers[loading].transform(paramFeatures)

                # Input and output size of the ANN
                inputSize = paramFeatures.shape[1]
                outputSize = stressLabels.shape[1]
                regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw][loading], L2_regularization)
    elif optimizeType == "hardening":
        for loading in loadings:
            # All loadings share the same parameters, but different stress values
            paramFeatures = np.array([list(dict(params).values()) for params in list(combined_loadings_interpolateCurves[loading].keys())])
            stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_loadings_interpolateCurves[loading].values())])
            # Normalizing the data
            paramFeatures = scalers[loading].transform(paramFeatures)
            # Input and output size of the ANN
            inputSize = paramFeatures.shape[1]
            outputSize = stressLabels.shape[1]
            regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
            regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw][loading], L2_regularization)