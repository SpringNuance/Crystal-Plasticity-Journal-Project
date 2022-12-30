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
    param_infos_GA_discrete = param_infos_GA_discrete_func(param_infos) # For GA discrete
    param_infos_GA_continuous = param_infos_GA_continuous_func(param_infos) # For GA continuous
    param_infos_BO = param_infos_BO_func(param_infos) # For both BO discrete and continuous
    param_infos_PSO_low = param_infos_PSO_low_func(param_infos)
    param_infos_PSO_high = param_infos_PSO_high_func(param_infos)
    # print("param_infos is:")
    # print(param_infos)
    # print("param_infos_GA_discrete is:")
    # print(param_infos_GA_discrete)
    # print("param_infos_GA_continuous is:")
    # print(param_infos_GA_continuous)
    # print("param_infos_BO is:")
    # print(param_infos_BO)
    # print("param_infos_PSO_low is:")
    # print(param_infos_PSO_low)
    # print("param_infos_PSO_high is:")
    # print(param_infos_PSO_high)
    time.sleep(30)
    # A mutual exclusion lock that ensures correct concurrency for printing and logging messages
    # messages is a list of message
    mutex = Lock()

    # A dictionary mutex shared by all processes of the form target curve -> False / True 
    # to ensure that the message is only printed once by all multiprocessing
    # checkMutex = Manager().dict()
    # for curveIndex in curveIndices:
    #     checkMutex[curveIndex] = False

    # def printOnceAndLog(messages, curveIndex):
    #     mutex.acquire()
    #     logPath = f"log/{material}/{CPLaw}/{CPLaw}{curveIndex}.txt"
    #     messages = list(map(lambda message: f"({CPLaw}{curveIndex}) {message}", messages))
    #     with open(logPath, 'a+') as logFile:
    #         logFile.writelines(messages)
    #     checkMutex[curveIndex] = True
    #     check = True
    #     for checkStatus in checkMutex:
    #         check = check and checkMutex[checkStatus]
    #     if check == True:
    #         for message in messages:
    #             print(message, end = '')
    #         for checkStatus in checkMutex:
    #             checkMutex[checkStatus] = False
    #     mutex.release()

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

    assert largeLinearYieldingDevGlobal > smallLinearYieldingDevGlobal, "largeLinearYieldingDev must be larger than smallLinearYieldingDev"
    assert largeLinearHardeningDevGlobal > smallLinearHardeningDevGlobal, "largeLinearHardeningDev must be larger than smallLinearHardeningDev"
    assert largeLinearHardeningDevGlobal >= smallLinearYieldingDevGlobal, "largeLinearHardeningDev must be larger than or equal smallLinearYieldingDev"

    assert largeNonlinearYieldingDevGlobal > smallNonlinearYieldingDevGlobal, "largeNonlinearYieldingDev must be larger than smallNonlinearYieldingDev"
    assert largeNonlinearHardeningDevGlobal > smallNonlinearHardeningDevGlobal, "largeNonlinearHardeningDev must be larger than smallNonlinearHardeningDev"
    assert largeNonlinearHardeningDevGlobal >= smallNonlinearYieldingDevGlobal, "largeNonlinearHardeningDev must be larger than or equal smallNonlinearYieldingDev"

    assert smallNonlinearYieldingDevGlobal > smallLinearYieldingDevGlobal, "smallNonlinearYieldingDev must be larger than smallLinearYieldingDev"
    assert smallNonlinearHardeningDevGlobal > smallLinearHardeningDevGlobal, "smallNonlinearHardeningDev must be larger than smallLinearHardeningDev"
    
    printAndLogAll(["Generating necessary directories\n"], curveIndices)
    printAndLogAll([f"The path to your main project folder is\n", f"{projectPath}\n\n"], curveIndices)

    ANNstatus = logInfo()
    printAndLogAll(ANNstatus, curveIndices)

    printAndLogAll(["\n" + 70 * "*" + "\n\n"], curveIndices)
    printAndLogAll([f"Step 0: Running initial simulations\n\n"], curveIndices)
    
    simUniversal = SIM(info)
    if method == "manual":
        printAndLogAll(["Starting initial simulations\n"], curveIndices)
        manualParams = np.load(f"manualParams/{material}/{CPLaw}/initial_params.npy", allow_pickle=True)
        tupleParams = manualParams[0:25] # <-- Run the parameters in small batches
        simUniversal.run_initial_simulations(tupleParams)
        printAndLogAll([f"Done. {len(tupleParams)} simulations completed."], curveIndices)
    elif method == "auto":
        if not os.path.exists(f"results/{material}/{CPLaw}/universal/initial_processCurves.npy"):
            printAndLogAll(["Starting initial simulations\n"], curveIndices)
            simUniversal.run_initial_simulations()
        initial_processCurves = np.load(f'results/{material}/{CPLaw}/universal/initial_processCurves.npy', allow_pickle=True).tolist()
        printAndLogAll([f"Done. {len(initial_processCurves['linear_uniaxial_RD'])} simulations completed\n"], curveIndices)
    
    initial_processCurvesGlobal = np.load(f'results/{material}/{CPLaw}/universal/initial_processCurves.npy', allow_pickle=True).tolist()
    initial_trueCurvesGlobal = np.load(f'results/{material}/{CPLaw}/universal/initial_trueCurves.npy', allow_pickle=True).tolist()
    reverse_initial_trueCurvesGlobal = reverseAsParamsToLoading(initial_trueCurvesGlobal, loadings)
    reverse_initial_processCurvesGlobal = reverseAsParamsToLoading(initial_processCurvesGlobal, loadings)
    np.save(f"results/{material}/{CPLaw}/universal/reverse_initial_trueCurves.npy", reverse_initial_trueCurvesGlobal)
    np.save(f"results/{material}/{CPLaw}/universal/reverse_initial_processCurves.npy", reverse_initial_processCurvesGlobal)
    getTargetCurves(material, CPLaw, curveIndices, loadings)

    printAndLogAll([f"Saving reverse and original initial true and process curves\n"], curveIndices)
    printAndLogAll([f"Finished preparing all target curves\n\n"], curveIndices)

    ###########################################
    # The main parallel optimizating function #
    ###########################################
    
    global parallelOptimization
    
    def parallelOptimization(curveIndex):

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
        for loading in loadings:
            exp_trueCurve = np.load(f'targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_true.npy', allow_pickle=True).tolist()
            exp_processCurve = np.load(f'targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_process.npy', allow_pickle=True).tolist()
            #print(loading)
            interpolatedStrain = interpolatingStrain(average_initialStrains[loading], exp_processCurve["stress"], yieldingPoints[CPLaw][loading], loading)                 
            interpolatedStress = interpolatingStress(exp_processCurve["strain"], exp_processCurve["stress"], interpolatedStrain, loading).reshape(-1) * convertUnit
            exp_interpolateCurve = {
                "strain": interpolatedStrain,
                "stress": interpolatedStress
            }

            exp_curves["true"][loading] = exp_trueCurve
            exp_curves["process"][loading] = exp_processCurve
            exp_curves["interpolate"][loading] = exp_interpolateCurve 
            np.save(f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_interpolate.npy", exp_curves["interpolate"][loading])
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

        #np.save(f"{iterationPath}/interpolateCurves.npy", interpolateCurves)
        #np.save(f"{iterationPath}/reverse_interpolateCurves.npy", reverse_interpolateCurves)

        messages.append(f"Saving interpolating {CPLaw}{curveIndex} target curves\n")
        messages.append("Saving initial interpolating curves and reverse initial interpolating curves\n")
        messages.append("Experimental and simulated curves preparation completed\n\n")

        messages.append(f"Curve {CPLaw}{curveIndex} info: \n")
        messages.append("-"*107 + "\n")
        messages.append(f"| Loading\t\t         | Exp σ_yield  | Large σ_yield sim range\t| Small σ_yield sim range\t|\n")
        messages.append("-"*107 + "\n")
        for loading in loadings:
            largeYieldingDevGlobal = largeLinearYieldingDevGlobal if loading == "linear_uniaxial_RD" else largeNonlinearYieldingDevGlobal
            smallYieldingDevGlobal = smallLinearYieldingDevGlobal if loading == "linear_uniaxial_RD" else smallNonlinearYieldingDevGlobal
            targetYieldStress = '{:.3f}'.format(round(exp_curves["interpolate"][loading]['stress'][1], 3))
            rangeSimLargeYieldBelow = '{:.3f}'.format(round(exp_curves["interpolate"][loading]["stress"][1] * (1 - largeYieldingDevGlobal * 0.01), 3))  
            rangeSimLargeYieldAbove = '{:.3f}'.format(round(exp_curves["interpolate"][loading]['stress'][1] * (1 + largeYieldingDevGlobal * 0.01), 3))
            rangeSimSmallYieldBelow = '{:.3f}'.format(round(exp_curves["interpolate"][loading]["stress"][1] * (1 - smallYieldingDevGlobal * 0.01), 3))  
            rangeSimSmallYieldAbove = '{:.3f}'.format(round(exp_curves["interpolate"][loading]['stress'][1] * (1 + smallYieldingDevGlobal * 0.01), 3))
            messages.append(f"| {loading}{tabsLoading[loading]}| {targetYieldStress} MPa\t| [{rangeSimLargeYieldBelow}, {rangeSimLargeYieldAbove}] MPa\t| [{rangeSimSmallYieldBelow}, {rangeSimSmallYieldAbove}] MPa\t|\n")
        messages.append("-"*107 + "\n\n")

        tupleParamsStresses = list(reverse_combine_interpolateCurves.items())[0:initial_length]

        sortedClosestHardening = list(sorted(tupleParamsStresses, key = lambda paramsStresses: fitnessHardeningAllLoadings(exp_curves["interpolate"], paramsStresses[1], loadings, weightsLoading, weightsHardening)))
        #time.sleep(180)
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

        messages.append(f"Parameter set of the closest simulation curve to the target curve {CPLaw}{curveIndex} is: \n")
        messages.append("-"*27 + "\n")
        for paramValue in default_params:
            exponent = param_infos[curveIndex][paramValue[0]]['exponent'] if param_infos[curveIndex][paramValue[0]]['exponent'] != "e0" else ""
            unit = paramsUnit[CPLaw][paramValue[0]]
            messages.append(f"| {paramValue[0]}{tabsParam[CPLaw][paramValue[0]]} | {paramValue[1]}{exponent} {unit}\n")
        messages.append("-"*27 + "\n")
        messages.append("Saving the default curves\n")
        messages.append(f"This parameter set will serve as default parameters for the 4 stage optimization of curve {CPLaw}{curveIndex}\n\n")
        
        printAndLog(messages, curveIndex)

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
        # ANN model selection from smallest test error 
        numberOfHiddenLayers = 2
        hiddenNodesFormula = "formula1"
        ANNOptimizer = "Adam"
        L2_regularization = 0.5
        learning_rate = 0.1
        #epochs = 2200
        loading_epochs = {
            "PH": {
                "linear_uniaxial_RD": 2200, 
                "nonlinear_biaxial_RD": 3200, 
                "nonlinear_biaxial_TD": 2200,     
                "nonlinear_planestrain_RD": 3600,     
                "nonlinear_planestrain_TD": 3000,     
                "nonlinear_uniaxial_RD": 3400, 
                "nonlinear_uniaxial_TD": 2000
            },
            "DB":{
                "linear_uniaxial_RD": 2800, 
                "nonlinear_biaxial_RD": 2200, 
                "nonlinear_biaxial_TD": 3000,     
                "nonlinear_planestrain_RD": 3200,     
                "nonlinear_planestrain_TD": 3600,     
                "nonlinear_uniaxial_RD": 2400, 
                "nonlinear_uniaxial_TD": 3400
            }

        }


        messages.append(f"ANN model: (parameters) -> (stress values at interpolating strain points)\n")
        messages.append(f"ANN configuration:\n")
        messages.append("-" * 43 + "\n")
        messages.append(f"| Number of hidden layers    | {numberOfHiddenLayers}\t\t|\n")
        messages.append(f"| Hidden layer nodes formula | {hiddenNodesFormula}\t|\n")
        messages.append(f"| ANN Optimizer              | {ANNOptimizer}\t|\n")
        messages.append(f"| Learning rate              | {learning_rate}\t|\n")
        messages.append(f"| L2 regularization term     | {L2_regularization}\t|\n")
        for loading in loadings:
            messages.append(f"| Epochs of {loading}{tabsLoading[loading]} | {loading_epochs[CPLaw][loading]}\t|\n")
        messages.append("-" * 43 + "\n\n")
        
        for loading in loadings:
            # All loadings share the same parameters, but different stress values
            paramFeatures = np.array([list(dict(params).values()) for params in list(combine_interpolateCurves[loading].keys())])
            stressLabels = np.array([strainstress["stress"] for strainstress in list(combine_interpolateCurves[loading].values())])

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
            # stressLabels_predict = regressors[loading].predictMany(paramFeatures)
            #messages.append("The model summary\n")

            #messages.append(regressors[loading].logModel() + "\n\n")
            #time.sleep(180)
            messages.append(f"Finish training ANN for loading {loading}\n")
            messages.append(f"Training MSE error: {trainingError[-1]}\n")

        messages.append(f"The number of combined interpolate curves is {len(combine_interpolateCurves[loading])}\n\n")
        messages.append(f"Finish training ANN for all loadings of curve {CPLaw}{curveIndex}\n\n")
        printAndLog(messages, curveIndex)

        # -------------------------------------------------------------------
        #   Step 3: Optimize the yielding parameters for the curves in parallel
        # -------------------------------------------------------------------        
        messages = []
        messages.append(70 * "*" + "\n")
        messages.append(f"Step 3: Assessment of number optimization stages and level of deviation percentage of curve {CPLaw}{curveIndex}\n\n")

        allParams = dict(default_params).keys()

        large_yieldingParams = list(filter(lambda param: general_param_info[param]["type"] == "large_yielding", allParams))
        small_yieldingParams = list(filter(lambda param: general_param_info[param]["type"] == "small_yielding", allParams))
        large_hardeningParams = list(filter(lambda param: general_param_info[param]["type"] == "large_hardening", allParams))
        small_hardeningParams = list(filter(lambda param: general_param_info[param]["type"] == "small_hardening", allParams))
        
        messages.append(f"The large yielding parameters are {large_yieldingParams}\n")
        messages.append(f"The small yielding parameters are {small_yieldingParams}\n")
        messages.append(f"The large hardening parameters are {large_hardeningParams}\n")
        messages.append(f"The small hardening parameters are {small_hardeningParams}\n\n")    
  
        if len(large_yieldingParams) == 0:
            messages.append("There are no large yielding parameters\n")
            messages.append("1st stage optimization not required\n")
        else:
            messages.append(f"There are {len(large_yieldingParams)} large yielding parameters\n")
            messages.append("1st stage optimization required\n")

        if len(small_yieldingParams) == 0:
            messages.append("There are no small yielding parameters\n")
            messages.append("2nd stage optimization not required\n")
        else:
            messages.append(f"There are {len(small_yieldingParams)} small yielding parameters\n")
            messages.append("2nd stage optimization required\n")
        
        if len(large_hardeningParams) == 0:
            messages.append("There are no large hardening parameters\n")
            messages.append("3rd stage optimization not required\n")
        else:
            messages.append(f"There are {len(large_hardeningParams)} large hardening parameters\n")
            messages.append("3rd stage optimization required\n")

        if len(small_hardeningParams) == 0:
            messages.append("There are no small hardening parameters\n")
            messages.append("4th stage optimization not required\n\n")
        else:
            messages.append(f"There are {len(small_hardeningParams)} small hardening parameters\n")
            messages.append("4th stage optimization required\n\n")

        if len(large_yieldingParams) != 0 and len(small_yieldingParams) == 0:
            messages.append("Because there are no small yielding parameters but there are large yielding parameters\n")
            messages.append("the large yielding deviation percentage is set as the small yielding deviation percentage\n")
            messages.append(f"largeLinearYieldingDev = {smallLinearYieldingDevGlobal}%\n\n")
            messages.append(f"largeNonlinearYieldingDev = {smallNonlinearYieldingDevGlobal}%\n\n")
            largeLinearYieldingDev = smallLinearYieldingDevGlobal    
            largeNonlinearYieldingDev = smallNonlinearYieldingDevGlobal
        else:
            largeLinearYieldingDev = largeLinearYieldingDevGlobal    
            largeNonlinearYieldingDev = largeNonlinearYieldingDevGlobal
        
        smallLinearYieldingDev = smallLinearYieldingDevGlobal
        smallNonlinearYieldingDev = smallNonlinearYieldingDevGlobal

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
        
        smallLinearHardeningDev = smallLinearHardeningDevGlobal
        smallNonlinearHardeningDev = smallNonlinearHardeningDevGlobal

        printAndLog(messages, curveIndex)

        info = {
            'param_info': param_infos[curveIndex],
            'CPLaw': CPLaw,
            'server': server,
            'curveIndex': curveIndex,
            'initialSims': initialSims,
            'projectPath': projectPath,
            'optimizerName': optimizerName,
            'material': material,
            'method': method,
            'searchingSpace': searchingSpace,
            'roundContinuousDecimals': roundContinuousDecimals,
            'loadings': loadings
        }
        
        sim = SIM(info)
        sim.fileNumber = iteration_length 

        info = {
            "param_info": param_infos[curveIndex],
            "param_info_GA_discrete": param_infos_GA_discrete[curveIndex],
            "param_info_GA_continuous": param_infos_GA_continuous[curveIndex],
            "param_info_BO": param_infos_BO[curveIndex],
            "param_infos_PSO_low": param_infos_PSO_low[curveIndex],
            "param_infos_PSO_high": param_infos_PSO_high[curveIndex],
            'loadings': loadings,
            "material": material,
            "CPLaw": CPLaw,
            "curveIndex": curveIndex,
            "optimizerName": optimizerName,
            "exp_curves": exp_curves,
            "convertUnit": convertUnit,
            "weightsYielding": weightsYielding,
            "weightsHardening": weightsHardening,
            "weightsLoading": weightsLoading,
            "regressors": regressors,
            "scalers": scalers,
            "searchingSpace": searchingSpace,   
            "roundContinuousDecimals": roundContinuousDecimals,
        }
        
        if optimizerName == "GA":
            fullOptimizerName = "Genetic Algorithm"
            optimizer = GA(info)
        elif optimizerName == "BO":
            fullOptimizerName = "Bayesian Optimization"
            optimizer = BO(info)
        elif optimizerName == "PSO":
            fullOptimizerName = "Particle Swarm Optimization"
            optimizer = PSO(info)

        messages = [f"The chosen optimizer is {fullOptimizerName}\n",
                    f"Starting the four stage optimization for curve {CPLaw}{curveIndex}\n\n"]

        printAndLog(messages, curveIndex)

        largeYieldingDevs = {}
        smallYieldingDevs = {}
        largeHardeningDevs = {}
        smallHardeningDevs = {}
        for loading in loadings:
            if loading == "linear_uniaxial_RD":
                largeYieldingDevs[loading] = largeLinearYieldingDev
                smallYieldingDevs[loading] = smallLinearYieldingDev
                largeHardeningDevs[loading] = largeLinearHardeningDev
                smallHardeningDevs[loading] = smallLinearHardeningDev
            else:
                largeYieldingDevs[loading] = largeNonlinearYieldingDev
                smallYieldingDevs[loading] = smallNonlinearYieldingDev
                largeHardeningDevs[loading] = largeNonlinearHardeningDev
                smallHardeningDevs[loading] = smallNonlinearHardeningDev 
        # ----------------------------------------------------------------------------
        #   Four optimization stage: Optimize the parameters for the curves in parallel 
        # ----------------------------------------------------------------------------
        deviationPercent = [largeYieldingDevs, smallYieldingDevs, largeHardeningDevs, smallHardeningDevs]
        deviationCondition = [insideYieldingDevAllLoadings, insideYieldingDevAllLoadings, insideHardeningDevAllLoadings, insideHardeningDevAllLoadings]
        optimize_params = [large_yieldingParams, small_yieldingParams, large_hardeningParams, small_hardeningParams]
        parameterType = ["large yielding", "small yielding", "large hardening", "small hardening"]
        optimize_type = ["yielding", "yielding", "hardening", "hardening"]
        ordinalUpper = ["First", "Second", "Third", "Fourth"]
        ordinalLower = ["first", "second", "third", "fourth"]
        ordinalNumber = ["1","2","3","4"]
        
        # Four optimization stages for-loop
        for stageNumber in range(0,4):
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
                
                messages.append("-"*27 + "\n")
                for param in stage_curves['parameters_dict']:
                    paramValue = stage_curves['parameters_dict'][param]
                    target =  "(target)" if param in optimize_params[stageNumber] else ""
                    exponent = param_infos[curveIndex][param]['exponent'] if param_infos[curveIndex][param]['exponent'] != "e0" else ""
                    unit = paramsUnit[CPLaw][param]
                    messages.append(f"| {param}{tabsParam[CPLaw][param]} | {paramValue}{exponent} {unit} {target}\n")
                messages.append("-"*27 + "\n")
                
                printAndLog(messages, curveIndex)
            else:
                if os.path.exists(f"{iterationPath}/stage{ordinalNumber[stageNumber]}_curves.npy"):
                    messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                    messages.append(f"The file stage{ordinalNumber[stageNumber]}_curves.npy detected, which means the {ordinalLower[stageNumber]} stage has finished\n")
                    messages.append(f"{ordinalUpper[stageNumber]} optimization stage finished\n\n")
                    
                    # Loading the existing curves
                    stage_curves = np.load(f"{iterationPath}/stage{ordinalNumber[stageNumber]}_curves.npy", allow_pickle=True).tolist()
                    
                    # Making the default curves of next stage as the current stage result curves
                    default_curves = copy.deepcopy(stage_curves)

                    messages.append("-"*27 + "\n")
                    for param in stage_curves['parameters_dict']:
                        paramValue = stage_curves['parameters_dict'][param]
                        target =  "(target)" if param in optimize_params[stageNumber] else ""
                        exponent = param_infos[curveIndex][param]['exponent'] if param_infos[curveIndex][param]['exponent'] != "e0" else ""
                        unit = paramsUnit[CPLaw][param]
                        messages.append(f"| {param}{tabsParam[CPLaw][param]} | {paramValue}{exponent} {unit} {target}\n")
                    messages.append("-"*27 + "\n")

                    printAndLog(messages, curveIndex)
                else: 
                    messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                    messages.append(f"Optimizing the parameters {', '.join(optimize_params[stageNumber])}\n")
                    messages.append(f"{ordinalUpper[stageNumber]} optimization stage starts\n\n")
                    printAndLog(messages, curveIndex)
                    
                    # Copying the default curves to stage curves as a template for optimization
                    stage_curves = copy.deepcopy(default_curves)

                    # The template default params for the optimizer to only optimize certain parameters
                    default_params = copy.deepcopy(default_curves["parameters_dict"])
                    
                    # Calculate whether the default curves satisfies all loadings
                    (allLoadingsSatisfied, notSatisfiedLoadings) = deviationCondition[stageNumber](exp_curves["interpolate"], stage_curves["interpolate"], loadings, deviationPercent[stageNumber])

                    while not allLoadingsSatisfied: 
                        # Increment the fileNumber by 1
                        sim.fileNumber += 1
                        
                        # The loadings that do not satisfy the deviation percentage
                        messages = []
                        messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                        messages.append(f"The loadings not satisfied are\n" + '\n'.join(notSatisfiedLoadings) + "\n\n")
                        printAndLog(messages, curveIndex)
                        
                        # Setting converging to False. 
                        converging = False
                        
                        #**************************************#
                        # Check whether the iteration simulation converges
                        while not converging:

                            # Initialize the optimizer
                            optimizer.initializeOptimizer(default_params, optimize_params[stageNumber], optimize_type[stageNumber])

                            # Running the optimizer
                            optimizer.run()

                            # Output the best params found by the optimizer 
                            # bestParams contains 3 keys: solution_dict, solution_tuple and solution_fitness
                            bestParams = optimizer.outputResult()

                            #**************************************#
                            # Check if these params is already in the results. If yes, make the optimizer runs again
                            while bestParams['solution_tuple'] in reverse_combine_interpolateCurves.keys():
                                messages = []
                                messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                                messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileNumber} ####\n")
                                messages.append(f"The best candidate parameters found by {optimizerName}\n")
                                        
                                messages.append("-"*27 + "\n")
                                for param in bestParams['solution_dict']:
                                    paramValue = bestParams['solution_dict'][param]
                                    target =  "(target)" if param in optimize_params[stageNumber] else ""
                                    exponent = param_infos[curveIndex][param]['exponent'] if param_infos[curveIndex][param]['exponent'] != "e0" else ""
                                    unit = paramsUnit[CPLaw][param]
                                    messages.append(f"| {param}{tabsParam[CPLaw][param]} | {paramValue}{exponent} {unit} {target}\n")
                                messages.append("-"*27 + "\n")

                                messages.append(f"Parameters already probed. {optimizerName} needs to run again to obtain new parameters\n")
                                messages.append(f"Retraining ANN with slightly different configurations to prevent repeated parameters\n")
                                messages.append(f"The number of combined interpolate curves is {len(combine_interpolateCurves['linear_uniaxial_RD'])}\n\n")
                                printAndLog(messages, curveIndex)

                                # In order to prevent repeated params, retraining ANN with slightly different configuration
                                
                                random_epochs = random.randint(3000, 4000)
                                random_learning_rate = random.uniform(0.005, 0.02)
                                random_L2_regularization = random.uniform(0.01, 1)
                                
                                for loading in loadings:
                                    # All loadings share the same parameters, but different stress values
                                    paramFeatures = np.array([list(dict(params).values()) for params in list(combine_interpolateCurves[loading].keys())])
                                    stressLabels = np.array([strainstress["stress"] for strainstress in list(combine_interpolateCurves[loading].values())])

                                    # Normalizing the data
                                    paramFeatures = scalers[loading].transform(paramFeatures)
                            
                                    # Input and output size of the ANN
                                    inputSize = paramFeatures.shape[1]
                                    outputSize = stressLabels.shape[1]
                                    
                                    regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                                    trainingError = regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, random_learning_rate, random_epochs, random_L2_regularization)
 
                                # Initialize the optimizer again
                                optimizer.initializeOptimizer(default_params, optimize_params[stageNumber], optimize_type[stageNumber])
                                
                                # Running the optimizer again
                                optimizer.run()

                                # Output the best params found by the optimizer again
                                bestParams = optimizer.outputResult()
                            
                            #**************************************#
                            # Outside the while loop of repeated parameters
                            
                            messages = []
                            messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                            messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileNumber} ####\n")
                            messages.append(f"The best candidate parameters found by {optimizerName}\n")
                                        
                            messages.append("-"*27 + "\n")
                            for param in bestParams['solution_dict']:
                                paramValue = bestParams['solution_dict'][param]
                                target =  "(target)" if param in optimize_params[stageNumber] else ""
                                exponent = param_infos[curveIndex][param]['exponent'] if param_infos[curveIndex][param]['exponent'] != "e0" else ""
                                unit = paramsUnit[CPLaw][param]
                                messages.append(f"| {param}{tabsParam[CPLaw][param]} | {paramValue}{exponent} {unit} {target}\n")
                            messages.append("-"*27 + "\n")

                            messages.append(f"This is new parameters. Fitness value of the best candidate parameters: {bestParams['solution_fitness']}\n\n")
                            
                            printAndLog(messages, curveIndex)
                            
                            #time.sleep(180)
                            messages = []
                            messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                            messages.append(f"Running iteration {sim.fileNumber} simulation\n\n")

                            printAndLog(messages, curveIndex)

                            # Running a single iteration simulation and extracting the iteration simulation result
                            converging, one_new_iteration_trueCurves, one_new_iteration_processCurves = sim.run_iteration_simulations(bestParams['solution_dict'])
                            
                            if not converging:
                                messages = []
                                messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                                messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileNumber} ####\n")
                                messages.append("Iteration simulation has not converged. Rerunning the optimizer to obtain another set of candidate parameters\n\n")
                                messages.append(f"Retraining ANN with slightly different configurations to prevent nonconverging parameters\n")
                                messages.append(f"The number of combined interpolate curves is {len(combine_interpolateCurves['linear_uniaxial_RD'])}\n\n")
                                
                                printAndLog(messages, curveIndex)
                                
                                # In order to prevent nonconverging params, retraining ANN with slightly different configuration
                                
                                random_learning_rate = random.uniform(0.005, 0.02)
                                random_L2_regularization = random.uniform(0.01, 1)
                                
                                for loading in loadings:
                                    # All loadings share the same parameters, but different stress values
                                    paramFeatures = np.array([list(dict(params).values()) for params in list(combine_interpolateCurves[loading].keys())])
                                    stressLabels = np.array([strainstress["stress"] for strainstress in list(combine_interpolateCurves[loading].values())])

                                    # Normalizing the data
                                    paramFeatures = scalers[loading].transform(paramFeatures)
                            
                                    # Input and output size of the ANN
                                    inputSize = paramFeatures.shape[1]
                                    outputSize = stressLabels.shape[1]
                                    
                                    regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                                    trainingError = regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, random_learning_rate, loading_epochs[CPLaw][loading], random_L2_regularization)
                        #**************************************#
                        # Outside the while loop of converging
                        messages = []
                        messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                        messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileNumber} ####\n")
                        messages.append("Iteration simulation has converged. Saving the one new iteration simulation curves\n\n")

                        printAndLog(messages, curveIndex)

                        # Update the iteration curves 
                        for loading in loadings:
                            iteration_trueCurves[loading].update(one_new_iteration_trueCurves[loading])
                            iteration_processCurves[loading].update(one_new_iteration_processCurves[loading])
                        
                        # Update the reverse iteration curves
                        reverse_iteration_trueCurves.update(reverseAsParamsToLoading(one_new_iteration_trueCurves, loadings))
                        reverse_iteration_processCurves.update(reverseAsParamsToLoading(one_new_iteration_processCurves, loadings))
                        
                        # Update the current candidate curves (stage_curves) and interpolate iteration curves
                        stage_curves["parameters_tuple"] = bestParams["solution_tuple"]
                        stage_curves["parameters_dict"] = bestParams["solution_dict"]
                        stage_curves["true"] = reverse_iteration_trueCurves[bestParams["solution_tuple"]]
                        stage_curves["process"] = reverse_iteration_processCurves[bestParams["solution_tuple"]]
                        for loading in loadings:
                            stage_curves["interpolate"][loading] = {
                                "strain": exp_curves["interpolate"][loading]["strain"], 
                                "stress": interpolatingStress(stage_curves["process"][loading]["strain"], stage_curves["process"][loading]["stress"]  * convertUnit, exp_curves["interpolate"][loading]["strain"], loading).reshape(-1)
                            }
                        
                        # Update iteration_interpolateCurves
                        for loading in loadings:
                            iteration_interpolateCurves[loading][stage_curves["parameters_tuple"]] = stage_curves["interpolate"][loading]
                        
                        # Update reverse_iteration_interpolateCurves
                        reverse_iteration_interpolateCurves[stage_curves["parameters_tuple"]] = stage_curves["interpolate"]
                        
                        # Update combine_interpolateCurves
                        for loading in loadings:
                            combine_interpolateCurves[loading][stage_curves["parameters_tuple"]] = stage_curves["interpolate"][loading]

                        # Update reverse_combine_interpolateCurves
                        reverse_combine_interpolateCurves[stage_curves["parameters_tuple"]] = stage_curves["interpolate"]
                        
                        # Saving the updated iteration curves
                        np.save(f"{iterationPath}/iteration_trueCurves.npy", iteration_trueCurves)
                        np.save(f"{iterationPath}/iteration_processCurves.npy", iteration_processCurves)
                        np.save(f"{iterationPath}/iteration_interpolateCurves.npy", iteration_interpolateCurves)
                        np.save(f"{iterationPath}/reverse_iteration_trueCurves.npy", reverse_iteration_trueCurves)
                        np.save(f"{iterationPath}/reverse_iteration_processCurves.npy", reverse_iteration_processCurves)
                        np.save(f"{iterationPath}/reverse_iteration_interpolateCurves.npy", reverse_iteration_interpolateCurves)

                        messages = []
                        messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                        messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileNumber} ####\n")
                        messages.append("Starting to retrain the ANN for all loadings\n")
                        messages.append(f"The number of combined interpolate curves is {len(combine_interpolateCurves['linear_uniaxial_RD'])}\n\n")
                     
                        printAndLog(messages, curveIndex)

                        # Retraining all the ANN
                        for loading in loadings:
                            # All loadings share the same parameters, but different stress values
                            paramFeatures = np.array([list(dict(params).values()) for params in list(combine_interpolateCurves[loading].keys())])
                            stressLabels = np.array([strainstress["stress"] for strainstress in list(combine_interpolateCurves[loading].values())])

                            # Normalizing the data
                            paramFeatures = scalers[loading].transform(paramFeatures)
                    
                            # Input and output size of the ANN
                            inputSize = paramFeatures.shape[1]
                            outputSize = stressLabels.shape[1]
                            
                            regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
                            trainingError = regressors[loading].train(paramFeatures, stressLabels, ANNOptimizer, learning_rate, loading_epochs[CPLaw][loading], L2_regularization)

                        messages = []
                        messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                        messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileNumber} ####\n")
                        messages.append(f"Finish training ANN for all loadings\n\n")
                                    
                        printAndLog(messages, curveIndex)
                        
                        # recalculate whether the one new iteration curves satisfies all loadings
                        (allLoadingsSatisfied, notSatisfiedLoadings) = deviationCondition[stageNumber](exp_curves["interpolate"], stage_curves["interpolate"], loadings, deviationPercent[stageNumber])

                    #**************************************#
                    # Outside the while loop of allLoadingsSatisfied
                    stage_curves["succeeding_iteration"] = sim.fileIndex

                    # Saving the result of the ith optimization stage 
                    np.save(f"{iterationPath}/stage{ordinalNumber[stageNumber]}_curves.npy", stage_curves)

                    # Making the default curves of next stage as the current stage result curves
                    default_curves = copy.deepcopy(stage_curves)

                    messages = []
                    messages.append(f"#### Stage {ordinalNumber[stageNumber]} ####\n")
                    messages.append(f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileNumber} ####\n")
                    messages.append(f"All loadings have successfully satisfied the {parameterType[stageNumber]} deviation\n")
                    messages.append(f"Succeeded iteration: {sim.fileNumber}\n")
                    messages.append(f"The {ordinalLower[stageNumber]} stage parameter solution is:\n")
                    
                    messages.append("-"*27 + "\n")
                    for param in stage_curves['parameters_dict']:
                        paramValue = stage_curves['parameters_dict'][param]
                        target =  "(target)" if param in optimize_params[stageNumber] else ""
                        exponent = param_infos[curveIndex][param]['exponent'] if param_infos[curveIndex][param]['exponent'] != "e0" else ""
                        unit = paramsUnit[CPLaw][param]
                        messages.append(f"| {param}{tabsParam[CPLaw][param]} | {paramValue}{exponent} {unit} {target}\n")
                    messages.append("-"*27 + "\n")
                               
                    messages.append(f"{ordinalUpper[stageNumber]} optimization stage finished\n\n")
                    
                    printAndLog(messages, curveIndex)
        
        # Outside the for-loop of 4 optimization stages

        messages = [f"All four optimization stages have successfully completed for curve {CPLaw}{curveIndex}\n"]
        messages.append("The final optimized set of parameters is:\n")             

        stage4_curves = np.load(f"{iterationPath}/stage4_curves.npy", allow_pickle=True).tolist()
        for param in stage4_curves['parameters_dict']:
            paramValue = stage4_curves['parameters_dict'][param]
            exponent = param_infos[curveIndex][param]['exponent'] if param_infos[curveIndex][param]['exponent'] != "e0" else ""
            unit = paramsUnit[CPLaw][param]
            messages.append(f"| {param}{tabsParam[CPLaw][param]} | {paramValue}{exponent} {unit} \n")
        messages.append("-"*27 + "\n\n")
        
        messages.append(f"Optimization for curve {CPLaw}{curveIndex} has finished\n")
        messages.append("Waiting for other curves to finish their optimization\n\n")

        printAndLog(messages, curveIndex)

    # This is where the program actualy starts to run the code. The whole part above is just in a function that optimizes one curve
    pool = multiprocessing.Pool() # Creating a pool of multithreads that run jobs in parallel
    pool.map(parallelOptimization, curveIndices) # Obtain the optimized fitting parameters for the target curves in parallel
    pool.close() # Closing the multiprocessing pool

    # ------------------------------
    #   Finalizing the optimization 
    # ------------------------------

    printAndLogAll(["\n" + 70 * "-" + "\n"], curveIndices)
    printAndLogAll(["Fitting parameter optimization for all target curves completed\n"], curveIndices)
    printAndLogAll(["Congratulations! Thank you for using the Crystal Plasticity Software\n"], curveIndices)

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


# Put this in stage 2, under for loading in loadings:
            # Stage 1 validation: choose model configurations
            # Choosing model configuration
            # Splitting the training data
            # test_ratio = 0.1
            # test_size = int(test_ratio * initial_length)
            # paramFeatures_test = paramFeatures[0:test_size]
            # paramFeatures_train = paramFeatures[test_size:total_length]
            # stressLabels_test = stressLabels[0:test_size]
            # stressLabels_train = stressLabels[test_size:total_length]
            # Normalizing the data
            # scalers[loading] = StandardScaler()
            # paramFeatures_train = scalers[loading].fit_transform(paramFeatures_train)
            # paramFeatures_test = scalers[loading].fit_transform(paramFeatures_test)
            # validation_errors = {}
            # training_errors = {}
            # epochs = 2000
            # if loading == "linear_uniaxial_RD":
            # messages = [f"\nLoading {loading}:\n"
            #             f"paramFeatures shape is {paramFeatures.shape}\n",
            #             f"stressLabels shape is {stressLabels.shape}\n"
            #             f"paramFeatures_train shape is {paramFeatures_train.shape}\n",
            #             f"stressLabels_train shape is {stressLabels_train.shape}\n"
            #             f"paramFeatures_test shape is {paramFeatures_test.shape}\n",
            #             f"stressLabels_test shape is {stressLabels_test.shape}\n"
            #             ]
            # for numberOfHiddenLayers in [1,2]: # Choose either 1 or 2
            #     for hiddenNodesFormula in ["formula1", "formula2"]: # Choose either 1 or 2
            #         for ANNOptimizer in ["Adam", "SGD"]:
            #             for learning_rate in [0.01, 0.005, 0.001]:
            #                 regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
            #                 #print(regressors[loading])         
            #                 trainingError = regressors[loading].train(paramFeatures_train, stressLabels_train, ANNOptimizer, learning_rate, epochs, L2_regularization)
            #                 stressLabels_predict = regressors[loading].predictMany(paramFeatures_test)
            #                 validationError = MSE_loss(stressLabels_predict, stressLabels_test)
            #                 #print()
            #                 #print(f"Finish training for {loading}\n")
            #                 print(f"MSE error: {trainingError[-1]}\n")
            #                 print(f"Number of epochs : {len(trainingError)}\n")
            #                 print(f"Validation loss: {validationError}")
            #                 training_errors[f"{str(numberOfHiddenLayers)}{hiddenNodesFormula}{ANNOptimizer}{str(learning_rate)}"] = trainingError
            #                 validation_errors[f"{str(numberOfHiddenLayers)}{hiddenNodesFormula}{ANNOptimizer}{str(learning_rate)}"] = validationError
            # np.save(f"optimizers/losses/training_errors.npy", training_errors)
            # np.save(f"optimizers/losses/validation_errors.npy", validation_errors)
            
            # Stage 2: Choose the number of epochs
            # Choosing number of epochs
            # Splitting the training data
            # test_ratio = 0.1
            # test_size = int(test_ratio * initial_length)
            # paramFeatures_test = paramFeatures[0:test_size]
            # paramFeatures_train = paramFeatures[test_size:total_length]
            # stressLabels_test = stressLabels[0:test_size]
            # stressLabels_train = stressLabels[test_size:total_length]
            # hiddenNodesFormula = "formula2"
            # numberOfHiddenLayers = 2
            # ANNOptimizer = "Adam"
            # learning_rate = 0.01
            # Normalizing the data
            # scalers[loading] = StandardScaler()
            # paramFeatures_train = scalers[loading].fit_transform(paramFeatures_train)
            # paramFeatures_test = scalers[loading].fit_transform(paramFeatures_test)
            # epochs_trainingErrors = {}
            # epochs_validationErrors = {}
            # epochs = 250
            # while epochs <= 7000:
            #     print(f"Number of epochs {epochs}")
            #     regressors[loading] = NeuralNetwork(inputSize, outputSize, hiddenNodesFormula, numberOfHiddenLayers).to(device)
            #     trainingError = regressors[loading].train(paramFeatures_train, stressLabels_train, ANNOptimizer, learning_rate, epochs, L2_regularization)
            #     stressLabels_predict = regressors[loading].predictMany(paramFeatures_test)
            #     validationError = MSE_loss(stressLabels_predict, stressLabels_test)
            #     print(f"Training error: {trainingError[-1]}\n")
            #     print(f"Validation error: {validationError}")
            #     epochs_trainingErrors[f"{str(epochs)}"] = trainingError[-1]
            #     epochs_validationErrors[f"{str(epochs)}"] = validationError
            #     epochs += 250
            # pathError = f"optimizers/losses/{ANNOptimizer}_lr{learning_rate}_{hiddenNodesFormula}_hidden{numberOfHiddenLayers}"
            # if not os.path.exists(pathError):
            #     os.mkdir(pathError)
            # np.save(f"{pathError}/epochs_trainingErrors.npy", epochs_trainingErrors)
            # np.save(f"{pathError}/epochs_validationErrors.npy", epochs_validationErrors)
            # print("Finished saving the data")
            # time.sleep(30)
            #prediction = regressors[loading](torch.FloatTensor([3,500,50,150,1,1,1,1,1,1]).reshape([1,-1]))
            #print(regressors[loading](torch.FloatTensor([3,500,50,150,1,1,1,1,1,1]).reshape([1,-1])))
            #print(regressors[loading].predict(np.array([3,500,50,150,1,1,1,1,1,1])))