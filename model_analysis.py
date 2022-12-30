###########################################################
#                                                         #
#         CRYSTAL PLASTICITY PARAMETER CALIBRATION        #
#   Tools required: DAMASK and Finnish Supercomputer CSC  #
#                                                         #
###########################################################

def main():

    # Loading the parameter information
    #                                                                                    numOfColumns, startingColumn, spacing, nrows
    getParamRanges(material, CPLaw, curveIndices, searchingSpace, searchingType, roundContinuousDecimals, 3, 9, 1)
    general_param_info = loadGeneralParam(material, CPLaw)
    param_infos = loadParamInfos(material, CPLaw, curveIndices)


    mutex = Lock()

    def printAndLog(messages, curveIndex):
        logPath = f"log/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}.txt"
        messages = list(map(lambda message: f"({CPLaw}{curveIndex}) {message}", messages))
        with open(logPath, 'a+') as logFile:
            logFile.writelines(messages)
        mutex.acquire()
        for message in messages:
            print(message, end = '')
        mutex.release()

    def printAndLogAll(messages, curveIndices):
        mutex.acquire()
        for curveIndex in curveIndices:
            logPath = f"log/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}.txt"
            with open(logPath, 'a+') as logFile:
                logFile.writelines(messages)
        for message in messages:
            print(message, end = '')
        mutex.release()


    info = {
        'param_info': general_param_info,
        'server': server,
        'hyperqueue': hyperqueue,
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
    printAndLogAll(configMessages, curveIndices)

    printAndLogAll(["Generating necessary directories\n"], curveIndices)
    printAndLogAll([f"The path to your main project folder is\n", f"{projectPath}\n\n"], curveIndices)

    ANNstatus = logInfo()
    printAndLogAll(ANNstatus, curveIndices)

    initial_processCurvesGlobal = np.load(f'results/{material}/{CPLaw}/universal/initial_processCurves.npy', allow_pickle=True).tolist()
    initial_trueCurvesGlobal = np.load(f'results/{material}/{CPLaw}/universal/initial_trueCurves.npy', allow_pickle=True).tolist()
    reverse_initial_trueCurvesGlobal = reverseAsParamsToLoading(initial_trueCurvesGlobal, loadings)
    reverse_initial_processCurvesGlobal = reverseAsParamsToLoading(initial_processCurvesGlobal, loadings)
    np.save(f"results/{material}/{CPLaw}/universal/reverse_initial_trueCurves.npy", reverse_initial_trueCurvesGlobal)
    np.save(f"results/{material}/{CPLaw}/universal/reverse_initial_processCurves.npy", reverse_initial_processCurvesGlobal)
    # Producing all target curves npy file
    getTargetCurves(material, CPLaw, curveIndices, expTypes, loadings)
    printAndLogAll([f"Saving reverse and original initial true and process curves\n"], curveIndices)
    printAndLogAll([f"Finished preparing all target curves\n\n"], curveIndices)

    # -------------------------------------------------------------------
    #   Step 1: Loading progress and preparing data
    # -------------------------------------------------------------------

    messages = []
    messages.append(70 * "*" + "\n")
    messages.append(f"Step 1: Loading progress and preparing data for curve {CPLaw}{curveIndex}\n\n")
    
    iterationPath = f"results/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}"
    initialPath = f"results/{material}/{CPLaw}/universal"
    # Loading initial curves
    initial_trueCurves = np.load(f'{initialPath}/initial_trueCurves.npy', allow_pickle=True).tolist()
    initial_processCurves = np.load(f'{initialPath}/initial_processCurves.npy', allow_pickle=True).tolist()
    
    # Loading reverse initial curves
    reverse_initial_trueCurves = np.load(f'{initialPath}/reverse_initial_trueCurves.npy', allow_pickle=True).tolist()
    reverse_initial_processCurves = np.load(f'{initialPath}/reverse_initial_processCurves.npy', allow_pickle=True).tolist()
    
    # Create combine curves
    combine_trueCurves = {}
    combine_processCurves = {}

    # Create reverse combine curves
    reverse_combine_trueCurves = {}
    reverse_combine_processCurves = {}

    if os.path.exists(f"{iterationPath}/iteration_processCurves.npy"):
        # Loading iteration curves
        iteration_trueCurves = np.load(f'{iterationPath}/iteration_trueCurves.npy', allow_pickle=True).tolist()
        iteration_processCurves = np.load(f'{iterationPath}/iteration_processCurves.npy', allow_pickle=True).tolist()
        iteration_interpolateCurves = np.load(f'{iterationPath}/iteration_interpolateCurves.npy', allow_pickle=True).tolist()
        
        # Loading reverse iteraion curves
        reverse_iteration_trueCurves = np.load(f'{iterationPath}/reverse_iteration_trueCurves.npy', allow_pickle=True).tolist()
        reverse_iteration_processCurves = np.load(f'{iterationPath}/reverse_iteration_processCurves.npy', allow_pickle=True).tolist()
        reverse_iteration_interpolateCurves = np.load(f'{iterationPath}/reverse_iteration_interpolateCurves.npy', allow_pickle=True).tolist()

        # Length of initial and iteration simulations
        initial_length = len(reverse_initial_processCurves)
        iteration_length = len(reverse_iteration_processCurves)
        
        messages.append(f"Curve {CPLaw}{curveIndex} status: \n")
        messages.append(f"{iteration_length} iteration simulations completed.\n")
        messages.append(f"{initial_length} initial simulations completed.\n")     
        messages.append(f"Total: {initial_length + iteration_length} simulations completed.")
        
        # Updating the combine curves with the initial simulations and iteration curves 
        for loading in loadings:
            combine_trueCurves[loading] = {}
            combine_processCurves[loading] = {}
            
            combine_trueCurves[loading].update(initial_trueCurves[loading])
            combine_processCurves[loading].update(initial_processCurves[loading])

            combine_trueCurves[loading].update(iteration_trueCurves[loading])
            combine_processCurves[loading].update(iteration_processCurves[loading])
        
        # Updating the reverse combine curves with the reverse initial simulations and reverse iteration curves 
        reverse_combine_trueCurves.update(reverse_initial_trueCurves)
        reverse_combine_processCurves.update(reverse_initial_processCurves)

        reverse_combine_trueCurves.update(reverse_iteration_trueCurves)
        reverse_combine_processCurves.update(reverse_iteration_processCurves)
    else:
        # Creating empty iteration curves
        iteration_trueCurves = {}
        iteration_processCurves = {}
        iteration_interpolateCurves = {}
        for loading in loadings:
            iteration_trueCurves[loading] = {}
            iteration_processCurves[loading] = {}
            iteration_interpolateCurves[loading] = {}

        # Creating empty reverse iteraion curves
        reverse_iteration_trueCurves = {}
        reverse_iteration_processCurves = {}
        reverse_iteration_interpolateCurves = {}

        # Updating the combine curves with only initial simulations 
        for loading in loadings:
            combine_trueCurves[loading] = {}
            combine_processCurves[loading] = {}
            
            combine_trueCurves[loading].update(initial_trueCurves[loading])
            combine_processCurves[loading].update(initial_processCurves[loading])

        # Updating the reverse combine curves with only reverse initial curves 
        reverse_combine_trueCurves.update(reverse_initial_trueCurves)
        reverse_combine_processCurves.update(reverse_initial_processCurves)

        initial_length = len(reverse_initial_processCurves)    
        iteration_length = 0

        messages.append(f"Curve {CPLaw}{curveIndex} status: \n")
        messages.append(f"{initial_length} initial simulations completed.\n")
        messages.append(f"No additional iteration simulations completed.\n")
    all_initialStrains = {}
    average_initialStrains = {}

    # Calculating average strain from initial simulations 
    for loading in loadings:
        all_initialStrains[loading] = np.array(list(map(lambda strainstress: strainstress["strain"], initial_processCurves[loading].values())))
        average_initialStrains[loading] = all_initialStrains[loading].mean(axis=0)

    exp_curves = {}
    exp_curves["true"] = {}
    exp_curves["process"] = {}
    exp_curves["interpolate"] = {}
    
    # Loading the target curve, calculating the interpolating curve and save the compact data of target curve
    # Loading the target curve, calculating the interpolating curve and save the compact data of target curve
    for loading in loadings:
        exp_trueCurve = np.load(f'targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_true.npy', allow_pickle=True).tolist()
        exp_processCurve = np.load(f'targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_process.npy', allow_pickle=True).tolist()
        # DAMASK simulated curve used as experimental curve
        if expTypes[curveIndex] == "D":
            interpolatedStrain = interpolatingStrain(average_initialStrains[loading], exp_processCurve["strain"], exp_processCurve["stress"], yieldingPoints[CPLaw][loading], loading)                 
            interpolatedStress = interpolatingStress(exp_processCurve["strain"], exp_processCurve["stress"], interpolatedStrain, loading).reshape(-1) * convertUnit
            exp_interpolateCurve = {
                "strain": interpolatedStrain,
                "stress": interpolatedStress
            }
        # Actual experimental curve (serrated flow curve and Swift Voce fitted curve)
        elif expTypes[curveIndex] == "E":
            interpolatedStrain = interpolatingStrain(average_initialStrains[loading], exp_processCurve["strain"], list(initial_processCurves[loading].values())[0]["stress"], yieldingPoints[CPLaw][loading], loading)                 
            interpolatedStress = interpolatingStress(exp_processCurve["strain"], exp_processCurve["stress"], interpolatedStrain, loading).reshape(-1) * convertUnit
            exp_interpolateCurve = {
                "strain": interpolatedStrain,
                "stress": interpolatedStress
            }
            #print(loading)
            #print("interpolatedStrain")
            #print(exp_interpolateCurve["strain"])
            #print("interpolatedStress")
            #print(exp_interpolateCurve["stress"])
            #print("\n")
        
        exp_curves["true"][loading] = exp_trueCurve
        exp_curves["process"][loading] = exp_processCurve
        exp_curves["interpolate"][loading] = exp_interpolateCurve 
        np.save(f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_interpolate.npy", exp_curves["interpolate"][loading])
    #time.sleep(180) 
    np.save(f"targets/{material}/{CPLaw}/{CPLaw}{curveIndex}_curves.npy", exp_curves)
       
    # Calculating the combine interpolated curves from combine curves and derive reverse_interpolate curves
    combine_interpolateCurves = {}
    for loading in loadings:
        combine_interpolateCurves[loading] = {}
        for paramsTuple in combine_processCurves[loading]:
            sim_strain = combine_processCurves[loading][paramsTuple]["strain"]
            sim_stress = combine_processCurves[loading][paramsTuple]["stress"]
            combine_interpolateCurves[loading][paramsTuple] = {}
            combine_interpolateCurves[loading][paramsTuple]["strain"] = exp_curves["interpolate"][loading]["strain"] 
            combine_interpolateCurves[loading][paramsTuple]["stress"] = interpolatingStress(sim_strain, sim_stress, exp_curves["interpolate"][loading]["strain"], loading).reshape(-1) * convertUnit

    reverse_combine_interpolateCurves = reverseAsParamsToLoading(combine_interpolateCurves, loadings)

    tupleParamsStresses = list(reverse_combine_interpolateCurves.items())[0:initial_length]

    sortedClosestHardening = list(sorted(tupleParamsStresses, key = lambda paramsStresses: fitnessHardeningAllLoadings(exp_curves["interpolate"], paramsStresses[1], loadings, weightsLoading, weightsHardening)))
    
    # Obtaining the default hardening parameters
    default_params = sortedClosestHardening[0][0]

    default_curves = {}
    default_curves["parameters_tuple"] = default_params
    default_curves["parameters_dict"] = dict(default_params)
    default_curves["true"] = reverse_initial_trueCurves[default_params]
    default_curves["process"] = reverse_initial_processCurves[default_params]
    default_curves["interpolate"] = reverse_combine_interpolateCurves[default_params]
    default_curves["succeeding_iteration"] = 0
    np.save(f"{iterationPath}/default_curves.npy", default_curves)

    # -------------------------------------------------------------------
    #   Step 2: Initialize the regressors for all loadings
    # -------------------------------------------------------------------

    messages = []
    messages.append(70 * "*" + "\n")
    messages.append(f"Step 2: Train the regressors for all loadings with the initial simulations of curve {CPLaw}{curveIndex}\n\n")

    # The ANN regressors for each loading condition
    regressors = {}
    # The regularization scaler for each loading condition
    scalers = {}
    
    for loading in loadings:
        
        if loading == "linear_uniaxial_RD":
        #if loading == "nonlinear_biaxial_RD":
        #if loading == "nonlinear_biaxial_TD":
        #if loading == "nonlinear_planestrain_RD":
        #if loading == "nonlinear_planestrain_TD":
        #if loading == "nonlinear_uniaxial_RD":
        #if loading == "nonlinear_uniaxial_TD": 
            paramFeatures = np.array([list(dict(params).values()) for params in list(combine_interpolateCurves[loading].keys())])
            stressLabels = np.array([strainstress["stress"] for strainstress in list(combine_interpolateCurves[loading].values())])

            # Input and output size of the ANN
            sampleSize = stressLabels.shape[0]
            inputSize = paramFeatures.shape[1]
            outputSize = stressLabels.shape[1]
            total_length = initial_length + iteration_length
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
            scalers[loading] = StandardScaler().fit(paramFeatures[0:total_length])
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


    messages.append(f"The number of combined interpolate curves is {len(combine_interpolateCurves[loading])}\n\n")
    messages.append(f"Finish training ANN for all loadings of curve {CPLaw}{curveIndex}\n\n")
    printAndLog(messages, curveIndex)




if __name__ == '__main__':
    # External libraries
    import pandas as pd
    import numpy as np
    import random
    import multiprocessing
    from multiprocessing import Manager
    from threading import Lock
    from modules.SIM import *
    from modules.preprocessing import *
    from modules.stoploss import *
    from modules.helper import *
    from optimizers.GA import *
    from optimizers.BO import *
    from optimizers.PSO import * 
    from optimizers.ANN import *
    import os
    from optimize_config import *
    from sklearn.preprocessing import StandardScaler
    main()

# python optimize.py
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu