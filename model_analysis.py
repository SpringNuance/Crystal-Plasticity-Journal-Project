###########################################################
#                                                         #
#         CRYSTAL PLASTICITY PARAMETER CALIBRATION        #
#   Tools required: DAMASK and Finnish Supercomputer CSC  #
#                                                         #
###########################################################

def main():

    initial_processCurves = {}
    initial_trueCurves = {}
    for loading in loadings:
        initial_processCurves[loading] = np.load(f'results/{material}/{CPLaw}/universal/{loading}/initial_processCurves.npy', allow_pickle=True).tolist()
        initial_trueCurves[loading] = np.load(f'results/{material}/{CPLaw}/universal/{loading}/initial_trueCurves.npy', allow_pickle=True).tolist()
    #print(initial_processCurves[example_loading][(('dipole', 5.04253), ('islip', 760.12006), ('omega', 6.15309), ('p', 0.41821), ('q', 1.68084), ('tausol', 155.32767), ('Qs', 3.72866), ('Qc', 1.703), ('v0', 38.09405), ('rho_e', 10.10506))])
    getTargetCurves(material, CPLaw, curveIndex, loadings)

    # -------------------------------------------------------------------
    #   Step 1: Loading progress and preparing data
    # -------------------------------------------------------------------

    messages = []
    messages.append(70 * "*" + "\n")
    messages.append(f"Step 1: Loading progress and preparing data for curve {CPLaw}{curveIndex}\n\n")
        
    all_initialStrains = {}
    all_initialStress = {}
    average_initialStrains = {}
    
    # Calculating average strain from initial simulations 
    for loading in loadings:
        all_initialStress[loading] = np.array(list(map(lambda strainstress: strainstress["stress"], initial_processCurves[loading].values())))
        all_initialStrains[loading] = np.array(list(map(lambda strainstress: strainstress["strain"], initial_processCurves[loading].values())))
        average_initialStrains[loading] = all_initialStrains[loading].mean(axis=0)
    #print(len(list(all_initialStrains[loading])))
    #print(all_initialStress[example_loading][0])
    exp_curves = {}
    exp_curves["true"] = {}
    exp_curves["process"] = {}
    exp_curves["interpolate"] = {}

    # Loading the target curve, calculating the interpolating curve and save the compact data of target curve
    for loading in loadings:
        exp_trueCurve = np.load(f'targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_true.npy', allow_pickle=True).tolist()
        exp_processCurve = np.load(f'targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_process.npy', allow_pickle=True).tolist()
        # DAMASK simulated curve used as experimental curve
        interpolatedStrain = interpolatingStrain(average_initialStrains[loading], exp_processCurve["strain"], all_initialStress[loading][0], yieldingPoints[CPLaw][loading], loading)                 
        interpolatedStress = interpolatingStress(exp_processCurve["strain"], exp_processCurve["stress"], interpolatedStrain, loading).reshape(-1)
        exp_interpolateCurve = {
            "strain": interpolatedStrain,
            "stress": interpolatedStress
        }
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
        for paramsTuple in initial_processCurves[loading]:
            sim_strain = initial_processCurves[loading][paramsTuple]["strain"]
            sim_stress = initial_processCurves[loading][paramsTuple]["stress"]
            combine_interpolateCurves[loading][paramsTuple] = {}
            combine_interpolateCurves[loading][paramsTuple]["strain"] = exp_curves["interpolate"][loading]["strain"] 
            combine_interpolateCurves[loading][paramsTuple]["stress"] = interpolatingStress(sim_strain, sim_stress, exp_curves["interpolate"][loading]["strain"], loading).reshape(-1)
    initial_length = len(list(combine_interpolateCurves[loading]))
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
        # if loading == "linear_uniaxial_TD":
        #if loading == "nonlinear_biaxial_RD":
        #if loading == "nonlinear_biaxial_TD":
        #if loading == "nonlinear_planestrain_RD":
        #if loading == "nonlinear_planestrain_TD":
        #if loading == "nonlinear_uniaxial_RD":
        #if loading == "nonlinear_uniaxial_TD": 
            paramFeatures = np.array([list(dict(params).values()) for params in list(combine_interpolateCurves[loading].keys())])
            stressLabels = np.array([strainstress["stress"] * 1e-6 for strainstress in list(combine_interpolateCurves[loading].values())])


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
            increments = 1000
            epochs = 2200
            while epochs <= 3000:
                print(f"Number of epochs {epochs}")
                #regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers, sampleSize).to(device)
                #trainingError = regressors[loading].train(paramFeatures_train, stressLabels_train, ANNOptimizer, learning_rate, epochs, L2_regularization)
                #stressLabels_predict = regressors[loading].predict(paramFeatures_test)
                #validationError = MSE_loss(stressLabels_predict, stressLabels_test)
                regressors[loading] = NeuralNetwork(outputSize, inputSize, hiddenNodesFormula, numberOfHiddenLayers, sampleSize).to(device)
                trainingError = regressors[loading].train(stressLabels_train, paramFeatures_train, ANNOptimizer, learning_rate, epochs, L2_regularization)

                paramFeatures_predict = regressors[loading].predict(stressLabels_test)
                validationError = MSE_loss(paramFeatures_predict, paramFeatures_test)
                predictParams = regressors[loading].predictOneDimension(stressLabels[0])
                print("True param")
                print(paramFeatures[0])
                print("Predicted param")
                print(np.squeeze(scalers[loading].inverse_transform(predictParams)))
                
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


if __name__ == '__main__':
    # External libraries
    import numpy as np
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