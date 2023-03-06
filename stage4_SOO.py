# External libraries
import os
import numpy as np
import optimize_config
import stage0_initial_simulations  
import stage1_prepare_data
import stage2_train_ANN
import stage3_stages_analysis
from modules.SIM_damask2 import *
from stage1_prepare_data import * 
from modules.preprocessing import *
from modules.stoploss import *
from modules.helper import *
from optimizers.GA import *
from optimizers.BO import *
from optimizers.PSO import *
from optimizers.ANN import *
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler


# Three optimization stages for-loop

def main_SOO(info, prepared_data, stages_data, trained_models):

    server = info['server']
    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
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

    initial_length = prepared_data['initial_length']
    iteration_length = prepared_data['iteration_length']
    exp_curve = prepared_data['exp_curve']
    initialResultPath = prepared_data['initialResultPath']
    iterationResultPath = prepared_data['iterationResultPath']
    stage_CurvesList = prepared_data['stage_CurvesList']

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

    regressors = trained_models["regressors"]
    scalers = trained_models["scalers"]

    deviationPercent = stages_data['deviationPercent']
    deviationCondition = stages_data['deviationCondition']
    optimizeParams = stages_data['optimizeParams']
    parameterType = stages_data['parameterType']
    optimizeType = stages_data['optimizeType']
    ordinalUpper = stages_data['ordinalUpper']
    ordinalLower = stages_data['ordinalLower']
    ordinalNumber = stages_data['ordinalNumber']

    sim = SIM(info)
    sim.fileIndex = iteration_length 

    if optimizerName == "GA":
        fullOptimizerName = "Genetic Algorithm"
        optimizer = GA(info)
    if optimizerName == "BO":
        fullOptimizerName = "Bayesian Optimization"
        optimizer = BO(info)
    if optimizerName == "PSO":
        fullOptimizerName = "Particle Swarm Optimization"
        optimizer = PSO(info)

    printLog(f"The chosen optimizer is {fullOptimizerName}\n", logPath)
    printLog(f"Starting the multiple stage optimization for curve {CPLaw}{curveIndex}\n\n", logPath)

    for stageNumber in range(0,3):
        printLog("\n" + 70 * "*" + "\n\n", logPath)
        printLog(f"{ordinalUpper[stageNumber]} optimization stage: Optimize the {parameterType[stageNumber]} parameters for the curve {CPLaw}{curveIndex}\n\n", logPath)
        
        if len(optimizeParams[stageNumber]) == 0:
            printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
            printLog(f"There are no {parameterType[stageNumber]} parameters to optimize\n", logPath)
            printLog(f"Saving {ordinalLower[stageNumber]} stage result curves as the default curves\n", logPath)
            printLog(f"{ordinalUpper[stageNumber]} optimization stage finished\n\n", logPath)
            # Copying the default curves to stage curves as a template for optimization
            stage_curves = copy.deepcopy(default_curves)
            # Saving the result of the ith optimization stage
            np.save(f"{iterationResultPath}/common/stage{ordinalNumber[stageNumber]}_curves.npy", stage_curves)

            # Making the default curves of next stage as the current stage result curves
            default_curves = copy.deepcopy(stage_curves)
            printLog("The current result parameters\n", logPath)
            printDictParametersClean(stage_curves['parameters_dict'],param_info,paramsUnit,CPLaw, logPath)
        else:
            if os.path.exists(f"{iterationResultPath}/common/stage{ordinalNumber[stageNumber]}_curves.npy"):
                printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                printLog(f"The file stage{ordinalNumber[stageNumber]}_curves.npy detected, which means the {ordinalLower[stageNumber]} stage has finished\n", logPath)
                printLog(f"{ordinalUpper[stageNumber]} optimization stage finished\n\n", logPath)
                # Loading the existing curves
                stage_curves = np.load(f"{iterationResultPath}/common/stage{ordinalNumber[stageNumber]}_curves.npy", allow_pickle=True).tolist()
                # Making the default curves of next stage as the current stage result curves
                default_curves = copy.deepcopy(stage_curves)
                printLog("The current result parameters\n", logPath)
                printDictParametersClean(stage_curves['parameters_dict'],param_info,paramsUnit, CPLaw, logPath)
            else: 
                printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                printLog(f"Optimizing the parameters {', '.join(optimizeParams[stageNumber])}\n", logPath)
                printLog(f"{ordinalUpper[stageNumber]} optimization stage starts\n\n", logPath)
                # Copying the default curves to stage curves as a template for optimization
                stage_curves = copy.deepcopy(default_curves)
                # The template default params for the optimizer to only optimize certain parameters
                default_params = copy.deepcopy(default_curves["parameters_dict"])
                # Calculate whether the default curves satisfies all loadings
                if optimizeType[stageNumber] == "yielding":
                    linearSatisfied = deviationCondition[stageNumber](exp_curve["interpolate"]["linear_uniaxial_RD"]["stress"], 
                                                                    stage_curves["interpolate"]["linear_uniaxial_RD"]["stress"], 
                                                                    exp_curve["interpolate"]["linear_uniaxial_RD"]["strain"], 
                                                                    deviationPercent[stageNumber]["linear_uniaxial_RD"])
                    allLoadingsSatisfied = linearSatisfied
                    notSatisfiedLoadings = ["linear_uniaxial_RD"]
                elif optimizeType[stageNumber] == "hardening":
                    (allLoadingsSatisfied, notSatisfiedLoadings) = deviationCondition[stageNumber](exp_curve["interpolate"], 
                                                                                                    stage_curves["interpolate"], 
                                                                                                    loadings, 
                                                                                                    deviationPercent[stageNumber])
                while not allLoadingsSatisfied: 
                    # Increment the fileIndex by 1
                    sim.fileIndex += 1
                    # The loadings that do not satisfy the deviation percentage
                    printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                    printLog(f"The loadings not satisfied are\n" + '\n'.join(notSatisfiedLoadings) + "\n\n", logPath)
                    # Setting converging to False. 
                    converging = False
                    #**************************************#
                    # Check whether the iteration simulation converges
                    while not converging:
                        # Initialize the optimizer
                        optimizer.initializeOptimizer(default_params, optimizeParams[stageNumber], optimizeType[stageNumber])
                        start = time.time()
                        # Running the optimizer
                        optimizer.run()
                        end = time.time()
                        # Output the best params found by the optimizer 
                        # bestParams contains 3 keys: solution_dict, solution_tuple and solution_fitness
                        bestParams = optimizer.outputResult()
                        #**************************************#
                        # Check if these params is already in the results. If yes, make the optimizer runs again
                        while bestParams['solution_tuple'] in reverse_combined_loadings_interpolateCurves.keys():
                            printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                            printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n", logPath)
                            printLog(f"Searching time by {optimizerName}: {end - start}s\n\n", logPath)
                            printLog(f"The best candidate parameters found by {optimizerName}\n", logPath)
                            printLog("The current result parameters\n", logPath)
                            printDictCalibratedParametersClean(bestParams['solution_dict'], optimizeParams, stageNumber, param_info, paramsUnit, CPLaw, logPath)

                            printLog(f"Parameters already probed. {optimizerName} needs to run again to obtain new parameters\n", logPath)
                            printLog(f"Retraining ANN with slightly different configurations to prevent repeated parameters\n", logPath)
                            printLog(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves['linear_uniaxial_RD'])}\n\n", logPath)
                            # In order to prevent repeated params, retraining ANN with slightly different configuration
                            if optimizeType[stageNumber] == "yielding":
                                # All loadings share the same parameters, but different stress values
                                paramFeatures = np.array([list(dict(params).values()) for params in list(combined_loadings_interpolateCurves["linear_uniaxial_RD"].keys())])
                                stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_loadings_interpolateCurves["linear_uniaxial_RD"].values())])

                                # Normalizing the data
                                paramFeatures = scalers["linear_uniaxial_RD"].transform(paramFeatures)
                        
                                # Input and output size of the ANN
                                inputSize = paramFeatures.shape[1]
                                outputSize = stressLabels.shape[1]
                                regressors["linear_uniaxial_RD"] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                                regressors["linear_uniaxial_RD"].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw]["linear_uniaxial_RD"], L2_regularization)
                            elif optimizeType[stageNumber] == "hardening":
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
                            # Initialize the optimizer again
                            optimizer.initializeOptimizer(default_params, optimizeParams[stageNumber], optimizeType[stageNumber])
                            start = time.time()
                            # Running the optimizer
                            optimizer.run()
                            end = time.time()
                            # Output the best params found by the optimizer again
                            bestParams = optimizer.outputResult()
                        #**************************************#
                        # Outside the while loop of repeated parameters
                        printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                        printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n", logPath)
                        printLog(f"Searching time by {optimizerName}: {end - start}s\n\n", logPath)
                        stringMessage = f"The best candidate parameters found by {optimizerName}\n"
                        printDictCalibratedParametersClean(bestParams['solution_dict'], optimizeParams, stageNumber, param_info, paramsUnit, CPLaw, logPath)
                        printLog(f"This is new parameters. Loss of the best candidate parameters: {bestParams['solution_fitness']}\n\n", logPath)                   
                        #time.sleep(180)
                        printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                        printLog(f"Running iteration {sim.fileIndex} simulation\n\n", logPath)
                        # Running a single iteration simulation and extracting the iteration simulation result
                        converging, one_new_iteration_trueCurves, one_new_iteration_processCurves = sim.run_iteration_simulations(bestParams['solution_dict'])
                        if not converging:
                            printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                            printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n", logPath)
                            printLog("Iteration simulation has not converged. Rerunning the optimizer to obtain another set of candidate parameters\n\n", logPath)
                            printLog(f"Retraining ANN with slightly different configurations to prevent nonconverging parameters\n", logPath)
                            printLog(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves['linear_uniaxial_RD'])}\n\n", logPath)
                            # In order to prevent nonconverging params, retraining ANN with slightly different configuration
                            if optimizeType[stageNumber] == "yielding":
                                # All loadings share the same parameters, but different stress values
                                paramFeatures = np.array([list(dict(params).values()) for params in list(combined_loadings_interpolateCurves["linear_uniaxial_RD"].keys())])
                                stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_loadings_interpolateCurves["linear_uniaxial_RD"].values())])
                                # Normalizing the data
                                paramFeatures = scalers["linear_uniaxial_RD"].transform(paramFeatures)
                                # Input and output size of the ANN
                                inputSize = paramFeatures.shape[1]
                                outputSize = stressLabels.shape[1]                            
                                regressors["linear_uniaxial_RD"] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                                regressors["linear_uniaxial_RD"].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw]["linear_uniaxial_RD"], L2_regularization)
                            elif optimizeType[stageNumber] == "hardening":
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
                    #**************************************#
                    # Outside the while loop of converging
                    printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                    printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n", logPath)
                    printLog("Iteration simulation has converged. Saving the one new iteration simulation curves\n\n", logPath)
                    # Update the iteration curves 
                    for loading in loadings:
                        iteration_loadings_trueCurves[loading].update(one_new_iteration_trueCurves[loading])
                        iteration_loadings_processCurves[loading].update(one_new_iteration_processCurves[loading])
                    # Update the reverse iteration curves
                    reverse_iteration_loadings_trueCurves.update(reverseAsParamsToLoading(one_new_iteration_trueCurves, loadings))
                    reverse_iteration_loadings_processCurves.update(reverseAsParamsToLoading(one_new_iteration_processCurves, loadings))
                    # Update the current candidate curves (stage_curves) and interpolate iteration curves
                    stage_curves = copy.deepcopy(stage_curves)
                    stage_curves["iteration"] = sim.fileIndex
                    stage_curves["stageNumber"] = stageNumber
                    stage_curves["parameters_tuple"] = bestParams["solution_tuple"]
                    stage_curves["parameters_dict"] = bestParams["solution_dict"]
                    stage_curves["true"] = reverse_iteration_loadings_trueCurves[bestParams["solution_tuple"]]
                    stage_curves["process"] = reverse_iteration_loadings_processCurves[bestParams["solution_tuple"]]
                    stage_curves["predicted_MSE"] = bestParams['solution_fitness']
                    
                    for loading in loadings:
                        stage_curves["interpolate"][loading] = {
                            "strain": exp_curve["interpolate"][loading]["strain"], 
                            "stress": interpolatingStress(stage_curves["process"][loading]["strain"], stage_curves["process"][loading]["stress"], exp_curve["interpolate"][loading]["strain"], loading).reshape(-1)
                        }
                    
                    MSE = calculateMSE(exp_curve["interpolate"], stage_curves["interpolate"], optimizeType[stageNumber], loadings,  weightsLoading, weightsYielding, weightsHardening)
                    stage_curves["MSE"] = MSE
                    stage_CurvesList.append(stage_curves)
                    np.save(f"{iterationResultPath}/common/stage_CurvesList.npy", stage_CurvesList)
                    
                    printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                    printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n", logPath)
                    printLog(f"The total weighted {optimizeType[stageNumber]} MSE of the iteration curve is {MSE['weighted_total_MSE']}\n\n", logPath)

                    # Update iteration_interpolateCurves
                    for loading in loadings:
                        iteration_loadings_interpolateCurves[loading][stage_curves["parameters_tuple"]] = stage_curves["interpolate"][loading]
                    
                    # Update reverse_iteration_interpolateCurves
                    reverse_iteration_loadings_interpolateCurves[stage_curves["parameters_tuple"]] = stage_curves["interpolate"]
                    
                    # Update combined_loadings_interpolateCurves
                    for loading in loadings:
                        combined_loadings_interpolateCurves[loading][stage_curves["parameters_tuple"]] = stage_curves["interpolate"][loading]

                    # Update reverse_combined_loadings_interpolateCurves
                    reverse_combined_loadings_interpolateCurves[stage_curves["parameters_tuple"]] = stage_curves["interpolate"]
                    
                    # Saving the updated iteration curves
                    np.save(f"{iterationResultPath}/common/iteration_trueCurves.npy", iteration_loadings_trueCurves)
                    np.save(f"{iterationResultPath}/common/iteration_processCurves.npy", iteration_loadings_processCurves)
                    np.save(f"{iterationResultPath}/common/iteration_interpolateCurves.npy", iteration_loadings_interpolateCurves)
                    np.save(f"{iterationResultPath}/common/reverse_iteration_trueCurves.npy", reverse_iteration_loadings_trueCurves)
                    np.save(f"{iterationResultPath}/common/reverse_iteration_processCurves.npy", reverse_iteration_loadings_processCurves)
                    np.save(f"{iterationResultPath}/common/reverse_iteration_interpolateCurves.npy", reverse_iteration_loadings_interpolateCurves)

                    printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                    printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n", logPath)
                    printLog("Starting to retrain the ANN for all loadings\n", logPath)
                    printLog(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves['linear_uniaxial_RD'])}\n\n", logPath)
                    
                    # Retraining all the ANN
                    if optimizeType[stageNumber] == "yielding":
                        paramFeatures = np.array([list(dict(params).values()) for params in list(combined_loadings_interpolateCurves["linear_uniaxial_RD"].keys())])
                        stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_loadings_interpolateCurves["linear_uniaxial_RD"].values())])

                        # Normalizing the data
                        paramFeatures = scalers["linear_uniaxial_RD"].transform(paramFeatures)
                
                        # Input and output size of the ANN
                        inputSize = paramFeatures.shape[1]
                        outputSize = stressLabels.shape[1]
                        
                        regressors["linear_uniaxial_RD"] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                        regressors["linear_uniaxial_RD"].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw]["linear_uniaxial_RD"], L2_regularization)
                    elif optimizeType[stageNumber] == "hardening":
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

                    printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                    printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n", logPath)
                    printLog(f"Finish training ANN for all loadings\n\n", logPath)
                                
                    # Calculate whether the default curves satisfies all loadings
                    if optimizeType[stageNumber] == "yielding":
                        linearSatisfied = deviationCondition[stageNumber](exp_curve["interpolate"]["linear_uniaxial_RD"]["stress"], 
                                                                        stage_curves["interpolate"]["linear_uniaxial_RD"]["stress"], 
                                                                        exp_curve["interpolate"]["linear_uniaxial_RD"]["strain"], 
                                                                        deviationPercent[stageNumber]["linear_uniaxial_RD"])
                        allLoadingsSatisfied = linearSatisfied
                        notSatisfiedLoadings = ["linear_uniaxial_RD"]
                    elif optimizeType[stageNumber] == "hardening":
                        (allLoadingsSatisfied, notSatisfiedLoadings) = deviationCondition[stageNumber](exp_curve["interpolate"], 
                                                                                                        stage_curves["interpolate"], 
                                                                                                        loadings, 
                                                                                                        deviationPercent[stageNumber])
                #**************************************#
                # Outside the while loop of allLoadingsSatisfied

                # Saving the result of the ith optimization stage 
                np.save(f"{iterationResultPath}/common/stage{ordinalNumber[stageNumber]}_curves.npy", stage_curves)

                # Making the default curves of next stage as the current stage result curves
                default_curves = copy.deepcopy(stage_curves)

                printLog(f"#### Stage {ordinalNumber[stageNumber]} ####\n", logPath)
                printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n", logPath)
                printLog(f"All loadings have successfully satisfied the deviation percentage\n", logPath)
                printLog(f"Succeeded iteration: {sim.fileIndex}\n", logPath)

                printLog(f"The {ordinalLower[stageNumber]} stage parameter solution is:\n", logPath)
                printDictParametersClean(stage_curves['parameters_dict'], param_info, paramsUnit, CPLaw, logPath)  
                printLog(f"{ordinalUpper[stageNumber]} optimization stage finished\n\n", logPath)
                
if __name__ == '__main__':
    info = optimize_config.main()
    prepared_data = stage1_prepare_data.main_prepareData(info)
    trained_models = stage2_train_ANN.main_trainANN(info, prepared_data)
    stages_data = stage3_stages_analysis.main_stagesAnalysis(info, prepared_data)
    main_SOO(info, prepared_data, stages_data, trained_models)