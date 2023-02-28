###########################################################
#                                                         #
#         CRYSTAL PLASTICITY PARAMETER CALIBRATION        #
#   Tools required: DAMASK and Finnish Supercomputer CSC  #
#                                                         #
###########################################################

def main():

    ####################################################
    # Step 0: Generating universal initial simulations #
    ####################################################

    initial_simulations.main()

    (initial_length, iteration_length, exp_curves, iterationPath, 
    combined_trueCurves, combined_processCurves, combined_interpolateCurves,  
    reverse_combined_trueCurves,  reverse_combined_processCurves,  reverse_combined_interpolateCurves,  
    iteration_trueCurves, iteration_processCurves, iteration_interpolateCurves, 
    reverse_iteration_trueCurves, reverse_iteration_processCurves, reverse_iteration_interpolateCurves, 
    stage_CurvesList) = prepare_data()
    
    # -------------------------------------------------------------------
    #   Step 2: Initialize the regressors for all loadings
    # -------------------------------------------------------------------

    messages = []
    messages.append(70 * "*" + "\n")
    messages.append(f"Step 2: Train the regressors for all loadings with the initial simulations of curve {CPLaw}{curveIndex}\n\n")
    messages.append(f"ANN model: (parameters) -> (stress values at interpolating strain points)\n")
    
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
    stringMessage += "\n\n"

    messages.append(stringMessage)
    
    start = time.time()
    for loading in loadings:
        # All loadings share the same parameters, but different stress values
        paramFeatures = np.array([list(dict(params).values()) for params in list(combined_interpolateCurves[loading].keys())])
        stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_interpolateCurves[loading].values())])

        # Normalizing the data
        scalers[loading] = StandardScaler().fit(paramFeatures[0:initial_length])
        paramFeatures = scalers[loading].transform(paramFeatures)

        # Input and output size of the ANN
        sampleSize = stressLabels.shape[0]
        inputSize = paramFeatures.shape[1]
        outputSize = stressLabels.shape[1]

        messages.append(f"({loading}) paramFeatures shape is {paramFeatures.shape}\n")
        messages.append(f"({loading}) stressLabels shape is {stressLabels.shape}\n")

        regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers, sampleSize).to(device)
        trainingError = regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw][loading], L2_regularization)

        messages.append(f"Finish training ANN for loading {loading}\n")
        messages.append(f"Training MSE error: {trainingError[-1]}\n\n")

    end = time.time()

    messages.append(f"The number of combined interpolate curves is {len(combined_interpolateCurves[loading])}\n")
    messages.append(f"Finish training ANN for all loadings of curve {CPLaw}{curveIndex}\n")
    messages.append(f"Total training time: {end - start}s\n\n")

    printList(messages, curveIndex)

    # -------------------------------------------------------------------
    #   Step 3: Optimize the yielding parameters for the curves in parallel
    # -------------------------------------------------------------------        
    messages = []
    messages.append(70 * "*" + "\n")
    messages.append(f"Step 3: Assessment of number optimization stages and level of deviation percentage of curve {CPLaw}{curveIndex}\n\n")

    allParams = dict(default_params).keys()

    yieldingParams = list(filter(lambda param: general_param_info[param]["type"] == "yielding", allParams))
    large_hardeningParams = list(filter(lambda param: general_param_info[param]["type"] == "large_hardening", allParams))
    small_hardeningParams = list(filter(lambda param: general_param_info[param]["type"] == "small_hardening", allParams))
    
    messages.append(f"The yielding parameters are {yieldingParams}\n")
    messages.append(f"The large hardening parameters are {large_hardeningParams}\n")
    messages.append(f"The small hardening parameters are {small_hardeningParams}\n\n")    

    if len(yieldingParams) == 0:
        messages.append("There are yielding parameters\n")
        messages.append("1st stage optimization not required\n")
    else:
        messages.append(f"There are {len(yieldingParams)} yielding parameters\n")
        messages.append("1st stage optimization required\n")
    
    if len(large_hardeningParams) == 0:
        messages.append("There are no large hardening parameters\n")
        messages.append("2nd stage optimization not required\n")
    else:
        messages.append(f"There are {len(large_hardeningParams)} large hardening parameters\n")
        messages.append("2nd stage optimization required\n")

    if len(small_hardeningParams) == 0:
        messages.append("There are no small hardening parameters\n")
        messages.append("3rd stage optimization not required\n\n")
    else:
        messages.append(f"There are {len(small_hardeningParams)} small hardening parameters\n")
        messages.append("3rd stage optimization required\n\n")

    if len(large_hardeningParams) != 0 and len(small_hardeningParams) == 0:
        messages.append("Because there are no small hardening parameters but there are large hardening parameters\n")
        messages.append("the large hardening deviation percentage is set as the small hardening deviation percentage\n")
        messages.append(f"largeLinearHardeningDev = {smallLinearHardeningDevGlobal}%\n\n")
        messages.append(f"largeNonlinearHardeningDev = {smallNonlinearHardeningDevGlobal}%\n\n")
        largeLinearHardeningDev = smallLinearHardeningDevGlobal    
        largeNonlinearHardeningDev = smallNonlinearHardeningDevGlobal
    else:
        largeLinearHardeningDev = largeLinearHardeningDevGlobal    
        largeNonlinearHardeningDev = largeNonlinearHardeningDevGlobal
    
    linearYieldingDev = linearYieldingDevGlobal
    smallLinearHardeningDev = smallLinearHardeningDevGlobal
    smallNonlinearHardeningDev = smallNonlinearHardeningDevGlobal

    printList(messages, curveIndex)
    
    sim = SIM(info)
    sim.fileIndex = iteration_length 

    
    if optimizerName == "NSGA":
        fullOptimizerName = "Non sorting genetic Algorithm"
        optimizer = GA(info)

    messages = [f"The chosen optimizer is {fullOptimizerName}\n",
                f"Starting the four stage optimization for curve {CPLaw}{curveIndex}\n\n"]

    printList(messages, curveIndex)

    yieldingDevs = {}
    largeHardeningDevs = {}
    smallHardeningDevs = {}
    for loading in loadings:
        if loading.startswith("linear"):
            yieldingDevs[loading] = linearYieldingDev
            largeHardeningDevs[loading] = largeLinearHardeningDev
            smallHardeningDevs[loading] = smallLinearHardeningDev
        else:
            largeHardeningDevs[loading] = largeNonlinearHardeningDev
            smallHardeningDevs[loading] = smallNonlinearHardeningDev 
    # ----------------------------------------------------------------------------
    #   Four optimization stage: Optimize the parameters for the curves in parallel 
    # ----------------------------------------------------------------------------
    deviationPercent = [yieldingDevs, largeHardeningDevs, smallHardeningDevs]
    deviationCondition = [insideYieldingDevLinear, insideHardeningDevAllLoadings, insideHardeningDevAllLoadings]
    optimize_params = [yieldingParams, large_hardeningParams, small_hardeningParams]
    parameterType = ["yielding", "yielding", "large hardening", "small hardening"]
    optimize_type = ["yielding", "hardening", "hardening"]
    ordinalUpper = ["First", "Second", "Third"]
    ordinalLower = ["first", "second", "third"]
    ordinalNumber = ["1","2","3"]
    
    # Four optimization stages for-loop
    for stageNumber in range(0,3):
        messages = []
        messages.append(70 * "*" + "\n")
        messages.append(f"{ordinalUpper[stageNumber]} optimization stage: Optimize the {parameterType[stageNumber]} parameters for the curve {CPLaw}{curveIndex}\n\n")
        
        if len(optimize_params[stageNumber]) == 0:
            messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
            messages.append(f"There are no {parameterType[stageNumber]} parameters to optimize\n")
            messages.append(f"Saving {ordinalLower[stageNumber]} stage result curves as the default curves\n")
            messages.append(f"{ordinalUpper[stageNumber]} optimization stage finished\n\n")
            
            # Copying the default curves to stage curves as a template for optimization
            stage_curves = copy.deepcopy(default_curves)
            
            # Saving the result of the ith optimization stage
            np.save(f"{iterationPath}/stage{ordinalNumber[stageNumber]}_curves.npy", stage_curves)

            # Making the default curves of next stage as the current stage result curves
            default_curves = copy.deepcopy(stage_curves)

            ########
            stringMessage = "The current result parameters\n"
            
            logTable = PrettyTable()
            logTable.field_names = ["Parameter", "Value"]
            for param in stage_curves['parameters_dict']:
                paramValue = stage_curves['parameters_dict'][param]
                exponent = general_param_info[param]['exponent'] if general_param_info[param]['exponent'] != "e0" else ""
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

            printList(messages, curveIndex)
        else:
            if os.path.exists(f"{iterationPath}/stage{ordinalNumber[stageNumber]}_curves.npy"):
                messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                messages.append(f"The file stage{ordinalNumber[stageNumber]}_curves.npy detected, which means the {ordinalLower[stageNumber]} stage has finished\n")
                messages.append(f"{ordinalUpper[stageNumber]} optimization stage finished\n\n")
                
                # Loading the existing curves
                stage_curves = np.load(f"{iterationPath}/stage{ordinalNumber[stageNumber]}_curves.npy", allow_pickle=True).tolist()
                
                # Making the default curves of next stage as the current stage result curves
                default_curves = copy.deepcopy(stage_curves)

                ########
                stringMessage = "The current result parameters\n"
                
                logTable = PrettyTable()
                logTable.field_names = ["Parameter", "Value"]
                for param in stage_curves['parameters_dict']:
                    paramValue = stage_curves['parameters_dict'][param]
                    target =  "(target)" if param in optimize_params[stageNumber] else ""
                    exponent = general_param_info[param]['exponent'] if general_param_info[param]['exponent'] != "e0" else ""
                    unit = paramsUnit[CPLaw][param]
                    paramString = f"{paramValue}"
                    if exponent != "":
                        paramString += exponent
                    if unit != "":
                        paramString += f" {unit}"
                    if target != "":
                        paramString += f" {target}"
                    logTable.add_row([param, paramString])

                stringMessage += logTable.get_string()
                stringMessage += "\n"
                messages.append(stringMessage)  
                ########

                printList(messages, curveIndex)
            else: 
                messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                messages.append(f"Optimizing the parameters {', '.join(optimize_params[stageNumber])}\n")
                messages.append(f"{ordinalUpper[stageNumber]} optimization stage starts\n\n")
                printList(messages, curveIndex)
                
                # Copying the default curves to stage curves as a template for optimization
                stage_curves = copy.deepcopy(default_curves)

                # The template default params for the optimizer to only optimize certain parameters
                default_params = copy.deepcopy(default_curves["parameters_dict"])
                
                # Calculate whether the default curves satisfies all loadings
                if optimize_type[stageNumber] == "yielding":
                    linearSatisfied = deviationCondition[stageNumber](exp_curves["interpolate"]["linear_uniaxial_RD"]["stress"], 
                                                                    stage_curves["interpolate"]["linear_uniaxial_RD"]["stress"], 
                                                                    exp_curves["interpolate"]["linear_uniaxial_RD"]["strain"], 
                                                                    deviationPercent[stageNumber]["linear_uniaxial_RD"])
                    allLoadingsSatisfied = linearSatisfied
                    notSatisfiedLoadings = ["linear_uniaxial_RD"]
                elif optimize_type[stageNumber] == "hardening":
                    (allLoadingsSatisfied, notSatisfiedLoadings) = deviationCondition[stageNumber](exp_curves["interpolate"], 
                                                                                                    stage_curves["interpolate"], 
                                                                                                    loadings, 
                                                                                                    deviationPercent[stageNumber])

                while not allLoadingsSatisfied: 
                    # Increment the fileIndex by 1
                    sim.fileIndex += 1
                    
                    # The loadings that do not satisfy the deviation percentage
                    messages = []
                    messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                    messages.append(f"The loadings not satisfied are\n" + '\n'.join(notSatisfiedLoadings) + "\n\n")
                    printList(messages, curveIndex)
                    
                    # Setting converging to False. 
                    converging = False
                    
                    #**************************************#
                    # Check whether the iteration simulation converges
                    while not converging:

                        # Initialize the optimizer
                        optimizer.initializeOptimizer(default_params, optimize_params[stageNumber], optimize_type[stageNumber])

                        start = time.time()

                        # Running the optimizer
                        optimizer.run()

                        end = time.time()

                        # Output the best params found by the optimizer 
                        # bestParams contains 3 keys: solution_dict, solution_tuple and solution_fitness
                        bestParams = optimizer.outputResult()

                        #**************************************#
                        # Check if these params is already in the results. If yes, make the optimizer runs again
                        while bestParams['solution_tuple'] in reverse_combined_interpolateCurves.keys():
                            messages = []
                            messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                            messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n")
                            messages.append(f"Searching time by {optimizerName}: {end - start}s\n\n")
                            messages.append(f"The best candidate parameters found by {optimizerName}\n")
                                    
                            ########
                            stringMessage = "The current result parameters\n"
                            
                            logTable = PrettyTable()
                            logTable.field_names = ["Parameter", "Value"]

                            for param in bestParams['solution_dict']:
                                paramValue = bestParams['solution_dict'][param]
                                target =  "(target)" if param in optimize_params[stageNumber] else ""
                                exponent = general_param_info[param]['exponent'] if general_param_info[param]['exponent'] != "e0" else ""
                                unit = paramsUnit[CPLaw][param]
                                logTable.add_row([param, f"{paramValue}{exponent} {unit} {target}"])

                            stringMessage += logTable.get_string()
                            stringMessage += "\n"
                            messages.append(stringMessage)  
                            ########

                            messages.append(f"Parameters already probed. {optimizerName} needs to run again to obtain new parameters\n")
                            messages.append(f"Retraining ANN with slightly different configurations to prevent repeated parameters\n")
                            messages.append(f"The number of combined interpolate curves is {len(combined_interpolateCurves['linear_uniaxial_RD'])}\n\n")
                            printList(messages, curveIndex)

                            # In order to prevent repeated params, retraining ANN with slightly different configuration
                        
                            
                            if optimize_type[stageNumber] == "yielding":
                                # All loadings share the same parameters, but different stress values
                                paramFeatures = np.array([list(dict(params).values()) for params in list(combined_interpolateCurves["linear_uniaxial_RD"].keys())])
                                stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_interpolateCurves["linear_uniaxial_RD"].values())])

                                # Normalizing the data
                                paramFeatures = scalers["linear_uniaxial_RD"].transform(paramFeatures)
                        
                                # Input and output size of the ANN
                                inputSize = paramFeatures.shape[1]
                                outputSize = stressLabels.shape[1]
                                
                                regressors["linear_uniaxial_RD"] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                                regressors["linear_uniaxial_RD"].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw]["linear_uniaxial_RD"], L2_regularization)
                            elif optimize_type[stageNumber] == "hardening":
                                for loading in loadings:
                                    # All loadings share the same parameters, but different stress values
                                    paramFeatures = np.array([list(dict(params).values()) for params in list(combined_interpolateCurves[loading].keys())])
                                    stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_interpolateCurves[loading].values())])

                                    # Normalizing the data
                                    paramFeatures = scalers[loading].transform(paramFeatures)
                            
                                    # Input and output size of the ANN
                                    inputSize = paramFeatures.shape[1]
                                    outputSize = stressLabels.shape[1]
                                    
                                    regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                                    regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw][loading], L2_regularization)

                            # Initialize the optimizer again
                            optimizer.initializeOptimizer(default_params, optimize_params[stageNumber], optimize_type[stageNumber])
                            
                            start = time.time()

                            # Running the optimizer
                            optimizer.run()

                            end = time.time()

                            # Output the best params found by the optimizer again
                            bestParams = optimizer.outputResult()
                        
                        #**************************************#
                        # Outside the while loop of repeated parameters
                        
                        messages = []
                        messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                        messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n")
                        messages.append(f"Searching time by {optimizerName}: {end - start}s\n\n")
                        
                        ########
                        stringMessage = f"The best candidate parameters found by {optimizerName}\n"
                        
                        logTable = PrettyTable()
                        logTable.field_names = ["Parameter", "Value"]

                        for param in bestParams['solution_dict']:
                            paramValue = bestParams['solution_dict'][param]
                            target =  "(target)" if param in optimize_params[stageNumber] else ""
                            exponent = general_param_info[param]['exponent'] if general_param_info[param]['exponent'] != "e0" else ""
                            unit = paramsUnit[CPLaw][param]
                            paramString = f"{paramValue}"
                            if exponent != "":
                                paramString += exponent
                            if unit != "":
                                paramString += f" {unit}"
                            if target != "":
                                paramString += f" {target}"
                            logTable.add_row([param, paramString])

                        stringMessage += logTable.get_string()
                        stringMessage += "\n"
                        messages.append(stringMessage)  
                        ########

                        messages.append(f"This is new parameters. Loss of the best candidate parameters: {bestParams['solution_fitness']}\n\n")
                        
                        time.sleep(30)
                        printList(messages, curveIndex)
                        
                        #time.sleep(180)
                        messages = []
                        messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                        messages.append(f"Running iteration {sim.fileIndex} simulation\n\n")

                        printList(messages, curveIndex)

                        # Running a single iteration simulation and extracting the iteration simulation result
                        converging, one_new_iteration_trueCurves, one_new_iteration_processCurves = sim.run_iteration_simulations(bestParams['solution_dict'])
                        
                        if not converging:
                            messages = []
                            messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                            messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n")
                            messages.append("Iteration simulation has not converged. Rerunning the optimizer to obtain another set of candidate parameters\n\n")
                            messages.append(f"Retraining ANN with slightly different configurations to prevent nonconverging parameters\n")
                            messages.append(f"The number of combined interpolate curves is {len(combined_interpolateCurves['linear_uniaxial_RD'])}\n\n")
                            
                            printList(messages, curveIndex)
                            
                            # In order to prevent nonconverging params, retraining ANN with slightly different configuration
                            
                            if optimize_type[stageNumber] == "yielding":
                                # All loadings share the same parameters, but different stress values
                                paramFeatures = np.array([list(dict(params).values()) for params in list(combined_interpolateCurves["linear_uniaxial_RD"].keys())])
                                stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_interpolateCurves["linear_uniaxial_RD"].values())])

                                # Normalizing the data
                                paramFeatures = scalers["linear_uniaxial_RD"].transform(paramFeatures)
                        
                                # Input and output size of the ANN
                                inputSize = paramFeatures.shape[1]
                                outputSize = stressLabels.shape[1]
                                
                                regressors["linear_uniaxial_RD"] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                                regressors["linear_uniaxial_RD"].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw]["linear_uniaxial_RD"], L2_regularization)
                            elif optimize_type[stageNumber] == "hardening":
                                for loading in loadings:
                                    # All loadings share the same parameters, but different stress values
                                    paramFeatures = np.array([list(dict(params).values()) for params in list(combined_interpolateCurves[loading].keys())])
                                    stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_interpolateCurves[loading].values())])

                                    # Normalizing the data
                                    paramFeatures = scalers[loading].transform(paramFeatures)
                            
                                    # Input and output size of the ANN
                                    inputSize = paramFeatures.shape[1]
                                    outputSize = stressLabels.shape[1]
                                    
                                    regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                                    regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw][loading], L2_regularization)
                    #**************************************#
                    # Outside the while loop of converging
                    messages = []
                    messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                    messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n")
                    messages.append("Iteration simulation has converged. Saving the one new iteration simulation curves\n\n")

                    printList(messages, curveIndex)

                    # Update the iteration curves 
                    for loading in loadings:
                        iteration_trueCurves[loading].update(one_new_iteration_trueCurves[loading])
                        iteration_processCurves[loading].update(one_new_iteration_processCurves[loading])
                    
                    # Update the reverse iteration curves
                    reverse_iteration_trueCurves.update(reverseAsParamsToLoading(one_new_iteration_trueCurves, loadings))
                    reverse_iteration_processCurves.update(reverseAsParamsToLoading(one_new_iteration_processCurves, loadings))
                    
                    # Update the current candidate curves (stage_curves) and interpolate iteration curves
                    stage_curves = copy.deepcopy(stage_curves)
                    stage_curves["iteration"] = sim.fileIndex
                    stage_curves["stageNumber"] = stageNumber
                    stage_curves["parameters_tuple"] = bestParams["solution_tuple"]
                    stage_curves["parameters_dict"] = bestParams["solution_dict"]
                    stage_curves["true"] = reverse_iteration_trueCurves[bestParams["solution_tuple"]]
                    stage_curves["process"] = reverse_iteration_processCurves[bestParams["solution_tuple"]]
                    stage_curves["predicted_MSE"] = bestParams['solution_fitness']
                    
                    for loading in loadings:
                        stage_curves["interpolate"][loading] = {
                            "strain": exp_curves["interpolate"][loading]["strain"], 
                            "stress": interpolatingStress(stage_curves["process"][loading]["strain"], stage_curves["process"][loading]["stress"], exp_curves["interpolate"][loading]["strain"], loading).reshape(-1)
                        }
                    
                    MSE = calculateMSE(exp_curves["interpolate"], stage_curves["interpolate"], optimize_type[stageNumber], loadings,  weightsLoading, weightsYielding, weightsHardening)
                    stage_curves["MSE"] = MSE
                    stage_CurvesList.append(stage_curves)
                    np.save(f"{iterationPath}/stage_CurvesList.npy", stage_CurvesList)
                    
                    messages = []
                    messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                    messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n")
                    messages.append(f"The total weighted {optimize_type[stageNumber]} MSE of the iteration curve is {MSE['weighted_total_MSE']}\n\n")

                    printList(messages, curveIndex)

                    # Update iteration_interpolateCurves
                    for loading in loadings:
                        iteration_interpolateCurves[loading][stage_curves["parameters_tuple"]] = stage_curves["interpolate"][loading]
                    
                    # Update reverse_iteration_interpolateCurves
                    reverse_iteration_interpolateCurves[stage_curves["parameters_tuple"]] = stage_curves["interpolate"]
                    
                    # Update combined_interpolateCurves
                    for loading in loadings:
                        combined_interpolateCurves[loading][stage_curves["parameters_tuple"]] = stage_curves["interpolate"][loading]

                    # Update reverse_combined_interpolateCurves
                    reverse_combined_interpolateCurves[stage_curves["parameters_tuple"]] = stage_curves["interpolate"]
                    
                    # Saving the updated iteration curves
                    np.save(f"{iterationPath}/iteration_trueCurves.npy", iteration_trueCurves)
                    np.save(f"{iterationPath}/iteration_processCurves.npy", iteration_processCurves)
                    np.save(f"{iterationPath}/iteration_interpolateCurves.npy", iteration_interpolateCurves)
                    np.save(f"{iterationPath}/reverse_iteration_trueCurves.npy", reverse_iteration_trueCurves)
                    np.save(f"{iterationPath}/reverse_iteration_processCurves.npy", reverse_iteration_processCurves)
                    np.save(f"{iterationPath}/reverse_iteration_interpolateCurves.npy", reverse_iteration_interpolateCurves)

                    messages = []
                    messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                    messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n")
                    messages.append("Starting to retrain the ANN for all loadings\n")
                    messages.append(f"The number of combined interpolate curves is {len(combined_interpolateCurves['linear_uniaxial_RD'])}\n\n")
                    
                    printList(messages, curveIndex)

                    # Retraining all the ANN
                    if optimize_type[stageNumber] == "yielding":
                        paramFeatures = np.array([list(dict(params).values()) for params in list(combined_interpolateCurves["linear_uniaxial_RD"].keys())])
                        stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_interpolateCurves["linear_uniaxial_RD"].values())])

                        # Normalizing the data
                        paramFeatures = scalers["linear_uniaxial_RD"].transform(paramFeatures)
                
                        # Input and output size of the ANN
                        inputSize = paramFeatures.shape[1]
                        outputSize = stressLabels.shape[1]
                        
                        regressors["linear_uniaxial_RD"] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                        regressors["linear_uniaxial_RD"].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw]["linear_uniaxial_RD"], L2_regularization)
                    elif optimize_type[stageNumber] == "hardening":
                        for loading in loadings:
                            # All loadings share the same parameters, but different stress values
                            paramFeatures = np.array([list(dict(params).values()) for params in list(combined_interpolateCurves[loading].keys())])
                            stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_interpolateCurves[loading].values())])

                            # Normalizing the data
                            paramFeatures = scalers[loading].transform(paramFeatures)
                    
                            # Input and output size of the ANN
                            inputSize = paramFeatures.shape[1]
                            outputSize = stressLabels.shape[1]
                            
                            regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                            regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw][loading], L2_regularization)

                    messages = []
                    messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                    messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n")
                    messages.append(f"Finish training ANN for all loadings\n\n")
                                
                    printList(messages, curveIndex)
                    
                    # Calculate whether the default curves satisfies all loadings
                    if optimize_type[stageNumber] == "yielding":
                        linearSatisfied = deviationCondition[stageNumber](exp_curves["interpolate"]["linear_uniaxial_RD"]["stress"], 
                                                                        stage_curves["interpolate"]["linear_uniaxial_RD"]["stress"], 
                                                                        exp_curves["interpolate"]["linear_uniaxial_RD"]["strain"], 
                                                                        deviationPercent[stageNumber]["linear_uniaxial_RD"])
                        allLoadingsSatisfied = linearSatisfied
                        notSatisfiedLoadings = ["linear_uniaxial_RD"]
                    elif optimize_type[stageNumber] == "hardening":
                        (allLoadingsSatisfied, notSatisfiedLoadings) = deviationCondition[stageNumber](exp_curves["interpolate"], 
                                                                                                        stage_curves["interpolate"], 
                                                                                                        loadings, 
                                                                                                        deviationPercent[stageNumber])
                #**************************************#
                # Outside the while loop of allLoadingsSatisfied

                # Saving the result of the ith optimization stage 
                np.save(f"{iterationPath}/stage{ordinalNumber[stageNumber]}_curves.npy", stage_curves)

                # Making the default curves of next stage as the current stage result curves
                default_curves = copy.deepcopy(stage_curves)

                messages = []
                messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileIndex} ####\n")
                messages.append(f"All loadings have successfully satisfied the deviation percentage\n")
                messages.append(f"Succeeded iteration: {sim.fileIndex}\n")

                ########
                stringMessage = f"The {ordinalLower[stageNumber]} stage parameter solution is:\n"
                
                logTable = PrettyTable()
                logTable.field_names = ["Parameter", "Value"]

                for param in stage_curves['parameters_dict']:
                    paramValue = stage_curves['parameters_dict'][param]
                    target =  "(target)" if param in optimize_params[stageNumber] else ""
                    exponent = general_param_info[param]['exponent'] if general_param_info[param]['exponent'] != "e0" else ""
                    unit = paramsUnit[CPLaw][param]
                    paramString = f"{paramValue}"
                    if exponent != "":
                        paramString += exponent
                    if unit != "":
                        paramString += f" {unit}"
                    if target != "":
                        paramString += f" {target}"
                    logTable.add_row([param, paramString])

                stringMessage += logTable.get_string()
                stringMessage += "\n"
                messages.append(stringMessage)  
                ########
                            
                messages.append(f"{ordinalUpper[stageNumber]} optimization stage finished\n\n")
                
                printList(messages, curveIndex)
    
    # Outside the for-loop of 4 optimization stages

    messages = [f"All four optimization stages have successfully completed for curve {CPLaw}{curveIndex}\n"]

    ########
    stringMessage = "The final optimized set of parameters is:\n"
    
    logTable = PrettyTable()
    logTable.field_names = ["Parameter", "Value"]

    stage4_curves = np.load(f"{iterationPath}/stage4_curves.npy", allow_pickle=True).tolist()

    for param in stage4_curves['parameters_dict']:
        paramValue = stage4_curves['parameters_dict'][param]
        exponent = general_param_info[param]['exponent'] if general_param_info[param]['exponent'] != "e0" else ""
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
    # External libraries
    import os
    import numpy as np
    import initial_simulations  
    import prepare_data
    from modules.SIM_damask2 import *
    from prepare_data import * 
    from modules.preprocessing import *
    from modules.stoploss import *
    from modules.helper import *
    from optimizers.GA import *
    from optimizers.ANN import *
    from prettytable import PrettyTable
    from optimize_config import *
    from sklearn.preprocessing import StandardScaler
    main()
