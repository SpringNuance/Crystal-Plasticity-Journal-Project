###########################################################
#                                                         #
#         CRYSTAL PLASTICITY PARAMETER CALIBRATION        #
#   Tools required: DAMASK and Finnish Supercomputer CSC  #
#                                                         #
###########################################################

def main_modelAnalysis(info, prepared_data):
    
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


    
    param_info_filtered = {}
    for parameter, info in param_info.items():
        if param_info[parameter]["optimized_target"]:
            param_info_filtered[parameter] = info

    #print(param_info)

    featureMatrixScaling = np.zeros((2, len(list(param_info_filtered.keys()))))
    powerList = np.zeros(len(list(param_info_filtered.keys())))
    #print(featureMatrixScaling)
    #print(param_info_filtered)
    for index, parameter in enumerate(list(param_info_filtered.keys())):
        # print(np.linspace(param_info[parameter]["low"], param_info[parameter]["high"], spacing - 1))
        # print(np.linspace(param_info[parameter]["low"], param_info[parameter]["high"], spacing - 1).shape)
        featureMatrixScaling[:, index] = np.array([param_info_filtered[parameter]["low"], param_info_filtered[parameter]["high"]])
        powerList[index] = param_info_filtered[parameter]["power"]
    # print(featureMatrixScaling)
    # print(featureMatrixScaling.shape)
    # print(powerList)
    # print(powerList.shape)
    #time.sleep(60)

    
    for loading in loadings:
        
        #if loading == "linear_uniaxial_RD":
        #if loading == "linear_uniaxial_TD":
        #if loading == "nonlinear_biaxial_RD":
        #if loading == "nonlinear_biaxial_TD":
        #if loading == "nonlinear_planestrain_RD":
        #if loading == "nonlinear_planestrain_TD":
        #if loading == "nonlinear_uniaxial_RD":
        if loading == "nonlinear_uniaxial_TD": 
            paramFeatures = np.array([list(dict(params).values()) for params in list(combined_loadings_interpolateCurves[loading].keys())])
            stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_loadings_interpolateCurves[loading].values())])


            # Input and output size of the ANN
            sampleSize = stressLabels.shape[0]
            inputSize = paramFeatures.shape[1]
            outputSize = stressLabels.shape[1]
            total_length = initial_length
            #total_length = initial_length
            print("The total number of curves is", total_length)
            test_ratio = 0.1
            test_size = int(test_ratio * total_length)
            print("Test size is ", test_size)
            paramFeatures_test = paramFeatures[0:test_size]
            paramFeatures_train = paramFeatures[test_size: total_length]
            stressLabels_test = stressLabels[0:test_size]
            stressLabels_train = stressLabels[test_size:total_length]
            
            # Normalizing the data
            scalers[loading] = CustomScaler(featureMatrixScaling, powerList)
            paramFeatures_train = scalers[loading].transform(paramFeatures_train)
            paramFeatures_test = scalers[loading].transform(paramFeatures_test)
            
            #Stage 1 validation: choose model configurations
            # validation_errors = {}
            # training_errors = {}
            # ANNOptimizer = "Adam"
            # epochs = 2000

            # for numberOfHiddenLayers in [1,2]: 
            #     for hiddenNodesFormula in ["formula1", "formula2", "formula3"]: 
            #         for learning_rate in [0.1, 0.05, 0.01]:
            #             for L2_regularization in [0.02, 0.1, 0.5]:
            #                 regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers, sampleSize).to(device)
            #                 trainingError = regressors[loading].train(paramFeatures_train, stressLabels_train, ANNOptimizer, learning_rate, epochs, L2_regularization)
            #                 stressLabels_predict = regressors[loading].predict(paramFeatures_test)
            #                 validationError = MSE_loss(stressLabels_predict, stressLabels_test)
            #                 print(f"numHidden {numberOfHiddenLayers} - hiddenForm {hiddenNodesFormula} - lr {learning_rate} - L2 {L2_regularization}")
            #                 print(f"Training error: {trainingError[-1]}")
            #                 print(f"Validation loss: {validationError}\n")
            #                 training_errors[f"numHid_{numberOfHiddenLayers}|{hiddenNodesFormula}|lr_{learning_rate}|L2_{L2_regularization}"] = trainingError
            #                 validation_errors[f"numHid_{numberOfHiddenLayers}|{hiddenNodesFormula}|lr_{learning_rate}|L2_{L2_regularization}"] = validationError
            # np.save(f"optimizers/losses/stage1/{loading}_training_errors.npy", training_errors)
            # np.save(f"optimizers/losses/stage1/{loading}_validation_errors.npy", validation_errors)
            # print("Finish training!")
            # time.sleep(60)

            # Stage 2: Choose the number of epochs
        
            ANNOptimizer = "Adam"
            hiddenNodesFormula = "formula1"
            numberOfHiddenLayers = 2
            L2_regularization = 0.5
            learning_rate = 0.1

            epochs_trainingErrors = {}
            epochs_validationErrors = {}
            increments = 100
            epochs = 100
            while epochs <= 3000:
                print(f"Number of epochs {epochs}")
                regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers, sampleSize).to(device)
                trainingError = regressors[loading].train(paramFeatures_train, stressLabels_train, ANNOptimizer, learning_rate, epochs, L2_regularization)
                stressLabels_predict = regressors[loading].predict(paramFeatures_test)
                validationError = MSE_loss(stressLabels_predict, stressLabels_test)
                # regressors[loading] = NeuralNetwork(outputSize, inputSize, hiddenNodesFormula, numberOfHiddenLayers, sampleSize).to(device)
                # trainingError = regressors[loading].train(stressLabels_train, paramFeatures_train, ANNOptimizer, learning_rate, epochs, L2_regularization)

                # paramFeatures_predict = regressors[loading].predict(stressLabels_test)
                # validationError = MSE_loss(paramFeatures_predict, paramFeatures_test)
                # predictParams = regressors[loading].predictOneDimension(stressLabels[0])
                # print("True param")
                # print(paramFeatures[0])
                # print("Predicted param")
                # print(np.squeeze(scalers[loading].inverse_transform(predictParams)))
                
                print(f"Training error: {trainingError[-1]}")
                print(f"Validation error: {validationError}\n")
                epochs_trainingErrors[f"{str(epochs)}"] = trainingError[-1]
                epochs_validationErrors[f"{str(epochs)}"] = validationError
                epochs += increments
            pathError = f"optimizers/losses/stage2/{ANNOptimizer}_{hiddenNodesFormula}_hiddenLayers{numberOfHiddenLayers}_L2{L2_regularization}_lr{learning_rate}/{CPLaw}"
            if not os.path.exists(pathError):
                os.makedirs(pathError)
            np.save(f"{pathError}/{loading}_epochs_trainingErrors.npy", epochs_trainingErrors)
            np.save(f"{pathError}/{loading}_epochs_validationErrors.npy", epochs_validationErrors)
            print("Finished saving the data")
            time.sleep(30)


    print(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves[loading])}\n\n")
    print(f"Finish training ANN for all loadings of curve {CPLaw}{curveIndex}\n\n")


if __name__ == '__main__':
    # External libraries
    import numpy as np
    from modules.SIM_damask2 import *
    from modules.preprocessing import *
    from modules.stoploss import *
    from modules.helper import *
    from optimizers.GA import *
    from optimizers.BO import *
    from optimizers.PSO import * 
    from optimizers.ANN import *
    from optimizers.scaler import * 
    import stage1_prepare_data 
    import os
    from optimize_config import *
    from sklearn.preprocessing import StandardScaler
    info = main_config()
    prepared_data = stage1_prepare_data.main_prepareData(info)
    main_modelAnalysis(info, prepared_data)

# python optimize.py
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu