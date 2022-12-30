###########################################################
#                                                         #
#         CRYSTAL PLASTICITY PARAMETER CALIBRATION        #
#   Tools required: DAMASK and Finnish Supercomputer CSC  #
#                                                         #
###########################################################

def main():

    #                                                                                    numOfColumns, startingColumn, spacing, nrows
    getParamRanges(material, CPLaw, curveIndices, searchingSpace, searchingType, roundContinuousDecimals, 3, 8, 1)
    general_param_info = loadGeneralParam(material, CPLaw)
    param_infos = loadParamInfos(material, CPLaw, curveIndices)
    param_infos_no_round = param_infos_no_round_func(param_infos) # For GA discrete
    param_infos_no_step_dict = param_infos_no_step_dict_func(param_infos_no_round) # For GA continuous
    param_infos_no_step_tuple = param_infos_no_step_tuple_func(param_infos_no_round) # For BO discrete and continuous

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
    simUniversal = SIM(info)
    if method == "manual":
        print("\nStarting initial simulations")
        manualParams = np.load(f"manualParams/{material}/{CPLaw}/initial_params.npy", allow_pickle=True)
        tupleParams = manualParams[475:500]
        simUniversal.run_initial_simulations(tupleParams)
    elif method == "auto":
        if not os.path.exists(f"results/{material}/{CPLaw}/universal/initial_params.npy"):
            print("\nStarting initial simulations")
            simUniversal.run_initial_simulations()
    print(f"Done. {len(tupleParams)} simulations completed.")


    #                                                                 numOfColumns, startingColumn, spacing, skiprows
    getTargetCurves(material, CPLaw, curveIndices, loadings, stressUnit, convertUnit, 2, 4, 3, numberOfParams + 2)

    mutex = Lock()
    # A mutual exclusion lock that ensures correct concurrency for printing and logging messages
    # messages is a list of message
    def printAndLog(messages, logPath):
        mutex.acquire()
        with open(logPath, 'a') as logFile:
            logFile.writelines(messages)
        for message in messages:
            print(message)
        mutex.release()
        
    # print("param_infos is:")
    # print(param_infos)
    # print("param_infos_no_round is:")
    # print(param_infos_no_round)
    # print("param_infos_no_step_tuple is:")
    # print(param_infos_no_step_tuple)
    print("param_infos_no_step_dict is:")
    print(param_infos_no_step_dict)

    ###########################################
    # The main parallel optimizating function #
    ###########################################
    global parallelOptimization
    
    def parallelOptimization(curveIndex):
        logPath = f"log/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}.txt"
        messages = ["--------------------------------\n",
                    f"Stage 1: Running initial simulations/Loading progress and preparing data for curve {CPLaw}{curveIndex}\n"]
        printAndLog(messages, logPath)
        
        info = {
            'param_info': param_infos[curveIndex],
            'CPLaw': CPLaw,
            'curveIndex': curveIndex,
            'initialSims': initialSims,
            'projectPath': projectPath,
            'optimizerName': optimizerName,
            'material': material,
            'method': method,
            'searchingSpace': searchingSpace,
            'roundContinuousDecimals': roundContinuousDecimals,
            'loading': loading
        }

        sim = SIM(info)
        initialPaths = []
        iterationPaths = []

        for loading in loadings:
            initialPath = f"results/{material}/{CPLaw}/universal/{loading}"
            initialPaths.append(initialPath)
            iterationPath = f"results/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}/{loading}" 
            iterationPaths.append(iterationPath)
            if not os.path.exists(iterationPath):
                os.makedirs(iterationPath)

        sim.initial_flowCurves = {}
        sim.initial_trueCurves = {}
        for loading in loadings:
            initialPath = f"results/{material}/{CPLaw}/universal/{loading}"
            initial_flow = np.load(f'{initialPath}/initial_flowCurves.npy', allow_pickle=True)
            initial_true = np.load(f'{initialPath}/initial_trueCurves.npy', allow_pickle=True)
            initial_flow = initial_flow.tolist() 
            initial_true = initial_true.tolist() 
            sim.initial_flowCurves[loading] = initial_flow
            sim.initial_flowCurves[loading] = initial_true
            if path.exists(f"{iteration_path}/iterations.npy"):
                iteration_path = f"results/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}"
                iteration_flow = np.load(f'{iteration_path}/iteration_flowCurves.npy', allow_pickle=True)
                iteration_true = np.load(f'{iteration_path}/iteration_trueCurves.npy', allow_pickle=True)
                sim.initial_flowCurves[loading].update(initial_flow)
                sim.initial_flowCurves[loading].update(iteration_true)
                messages = [f"Curve {CPLaw}{curveIndex} status: \n",
                            f"{len(iteration_flow)} {loading} iteration simulations completed.\n",   
                            f"{len(sim.initial_flowCurves[loading])} {loading} initial simulations completed.\n"     
                            f"Total: {len(sim.initial_flowCurves[loading])} {loading} simulations completed."]
                printAndLog(messages, logPath)
            else:
                messages = [f"Curve {CPLaw}{curveIndex} status: \n",
                            f"No additional {loading} iteration simulations completed.\n",
                            f"{len(sim.initial_flowCurves[loading])} {loading} initial simulations completed.\n"]
                printAndLog(messages, logPath)
            simStrains = list(map(lambda x: x[0], list(initial_flow.values())))
            average_strain = np.array(simStrains).mean(axis=0)
            sim.fileNumber = 1
        
            exp_curve = pd.read_csv(f'targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_flow.csv')   
            exp_stress = exp_curve.iloc[:,0] # Getting the experimental stress
            exp_strain = exp_curve.iloc[:,1] # Getting the experimental strain
            # The common strain points of experimental and simulated curves will be 
            # lying between 0.002 (strain of yield stress) and the maximum strain value of experimental curve 
    
            interpolated_strain = calculateInterpolatingStrains(simStrains, exp_strain, average_strain, yieldingPoints["linear_uniaxial_RD"]) 
            exp_stress = interpolatedStressFunction(exp_stress, exp_strain, interpolated_strain).reshape(-1) * convertUnit
            exp_strain = interpolated_strain
   
        print("Experimental and simulated curves preparation completed")

        # -------------------------------------------------------------------
        #   Stage 3 and 4: Optimize the yielding and hardening parameters for the curves in parallel
        # -------------------------------------------------------------------
        print("--------------------------------")
        print("Stage 3 and 4: Optimize the yielding and hardening parameters for the curves in parallel")

        # -------------------------------------------------------------------
        #   Stage 2: Initialize the regressors
        # -------------------------------------------------------------------
        # Input layer of fitting parameters (4 for PH and 6 for DB)
        print("--------------------------------")
        print("Stage 2: Initialize and train the regressors with the initial simulations ")
        X = np.array(list(sim.initial_flowCurves.keys()))
        # Output layer of the size of the interpolated stresses
        y = np.array([interpolatedStressFunction(simStress, simStrain, exp_strain) * convertUnit for (simStrain, simStress) in sim.initial_flowCurves.values()])
        inputSize = X.shape[1]
        outputSize = y.shape[1]
        hiddenSize1 = round((1/3) * (inputSize + outputSize))
        hiddenSize2 = round((2/3) * (inputSize + outputSize))
        regressor = MLPRegressor(hidden_layer_sizes=[hiddenSize1, hiddenSize2],activation='relu', alpha=0.001, solver='adam', max_iter=100000, shuffle=True)
        regressor = regressor.fit(X,y)
        y_predict = regressor.predict(X)
        val_error = mean_squared_error(y,y_predict)
        print(f"MSE validation error of curve {CPLaw}{curveIndex}: {val_error}")
        print(f"Finish training the regressor for curve {CPLaw}{curveIndex}\n")

        info = {
            "param_info": param_infos[curveIndex],
            "param_info_no_round": param_infos_no_round[curveIndex],
            "param_info_no_step_tuple": param_infos_no_step_tuple[curveIndex],
            "param_info_no_step_dict": param_infos_no_step_dict[curveIndex],
            "exp_stress": exp_stress,
            "exp_strain": exp_strain,
            "regressor": regressor,
            'loading': loading,
            "material": material,
            "CPLaw": CPLaw,
            "curveIndex": curveIndex,
            "optimizerName": optimizerName,
            "yieldStressDev": yieldStressDev,
            "hardeningDev": hardeningDev,
            "convertUnit": convertUnit,
            "weightsYield": weightsYield,
            "weightsHardening": weightsHardening,
            "numberOfParams": numberOfParams,
            "searchingSpace": searchingSpace,   
            "roundContinuousDecimals": roundContinuousDecimals,
            "param_infos": param_infos,
            "param_infos_no_round": param_infos_no_round,
            "param_info_no_step_tuple": param_infos_no_step_tuple,
            "param_info_no_step_dict": param_infos_no_step_dict,
        }

        if optimizerName == "GA": 
            optimizer = GA(info)
        elif optimizerName == "BO":
            optimizer = BO(info)    
        #elif optimizerName == "PSO":
        #    optimizer = PSO(info) 
        
        y = np.array([interpolatedStressFunction(simStress, simStrain, exp_strain) * convertUnit for (simStrain, simStress) in sim.initial_flowCurves.values()])
        # Obtaining the default hardening parameters
        targetYieldStress = exp_stress[1]
        zipParamsStress = list(zip(list(sim.initial_flowCurves.keys()), y))
        shiftedToTargetYieldStress = list(map(lambda paramZipsimStress: (paramZipsimStress[0], paramZipsimStress[1] + (targetYieldStress - paramZipsimStress[1][1])), zipParamsStress))
        sortedClosestHardening = list(sorted(shiftedToTargetYieldStress, key=lambda pairs: fitness_hardening(exp_stress, pairs[1], exp_strain, weightsHardening["wh1"], weightsHardening["wh2"], weightsHardening["wh3"], weightsHardening["wh3"])))
        default_hardening_params = sortedClosestHardening[0][0]
        default_hardening_params = tupleOrListToDict(default_hardening_params, CPLaw)
        
        rangeSimYield = (exp_stress[1] * (1 - yieldStressDev * 0.01), exp_stress[1] * (1 + yieldStressDev * 0.01)) 
        
        messages = [f"\nCurve {CPLaw}{curveIndex} info: ",
                    f"The target yield stress is {exp_stress[1]} MPa",
                    f"The simulated yield stress should lie in the range of {rangeSimYield} MPa",
                    f"Maximum allowed deviation: {exp_stress[1] * yieldStressDev * 0.01} MPa",
                    f"The default hardening parameters are: ",
                    f"{str(default_hardening_params)}"]
        
        printAndLog(messages, logPath)
        if optimizerName == "GA":
            optimizer = GA(info)
        if optimizerName == "BO":
            optimizer = BO(info)
        if optimizerName == "PSO":
            optimizer = PSO(info)

        optimizer.InitializeFirstStageOptimizer(default_hardening_params)
        
        while not insideYieldStressDev(exp_stress, y[-1], yieldStressDev):            
            optimizer.FirstStageRun()
            partialResults = optimizer.FirstStageOutputResult()
            converging = False
            while tuple(partialResults['solution']) in sim.initial_flowCurves.keys() or not converging:
                if tuple(partialResults['solution']) in sim.initial_flowCurves.keys():
                    messages = [f"(Curve {CPLaw}{curveIndex}) The predicted solution is:",
                                str(partialResults["solution_dict"]),
                                "Parameters already probed. Algorithm needs to run again to obtain new parameters"]
                    printAndLog(messages)
                    optimizer.FirstStageRun()
                    partialResults = optimizer.FirstStageOutputResult()
                    continue

                partialResult = partialResults['solution_dict']

                converging = sim.run_single_test(tuple(partialResults['solution']))
                if converging == False:
                    continue
    
                X = np.array(list(sim.simulations.keys()))
                y = np.array([interpolatedStressFunction(simStress, simStrain, exp_strain) * convertUnit for (simStrain, simStress) in sim.initial_flowCurves.values()])
                regressor.fit(X, y)
 
            messages = [f"#### (Curve {CPLaw}{curveIndex}) Iteration {sim.fileNumber + 1}  ####",
                        f"Parameters of the best partial solution : {partialResults['solution_dict']}",
                        f"Fitness value of the best solution = {partialResults['solution_fitness']}",
                        f"Fitness given by the regressor: {partialResults['fitness']}",
                        f"The simulated yield stress:", {y[-1][0]},"MPa"]
            printAndLog(messages)
        
        print("--------------------------------")
        print(f"(Curve {CPLaw}{curveIndex}) Yield stress parameters optimization completed")
        print("The partial parameter solution is: ")
        print(partialResult)
        print("Succeeded iteration:", sim.fileNumber)
        np.save(f'results/{material}/{CPLaw}{curveIndex}_{optimizerName}/partial_result.npy', partialResult)
        
        print("The partial result: ")
        print(partialResult)
        y = np.array([interpolatedStressFunction(simStress, simStrain, exp_strain) * convertUnit for (simStrain, simStress) in sim.simulations.values()])
        fullResult = list(sim.simulations.keys())[-1]
        print("The initial candidate full result: ")
        print(fullResult)
        optimizer.InitializeSecondStageOptimizer(partialResult)
        # Iterative optimization.
        while not insideHardeningDev(exp_stress, y[-1], hardeningDev):
            print("#### Iteration", sim.fileNumber + 1, "####")
            optimizer.SecondStageRun()
            fullResults = optimizer.SecondStageOutputResult()
            while tuple(fullResults['solution']) in sim.simulations.keys():
                print("The predicted solution is:")
                print(fullResults["solution_dict"])
                print("Parameters already probed. Algorithm needs to run again to obtain new parameters")
                optimizer.SecondStageRun()
                fullResults = optimizer.SecondStageOutputResult()
            optimizer.PrintResult(fullResults)
            # Wait a moment so that you can check the parameters predicted by the algorithm 
            time.sleep(20)
            fullResult = fullResults['solution_dict']
            sim.run_single_test(tuple(fullResults['solution']))
            X = np.array(list(sim.simulations.keys()))
            y = np.array([interpolatedStressFunction(simStress, simStrain, exp_strain) * convertUnit for (simStrain, simStress) in sim.simulations.values()])
            regressor.fit(X, y)
            loss = sqrt(mean_squared_error(y[-1], exp_stress))
            print(f"RMSE LOSS = {loss}")
        np.save(f'results/{material}/{CPLaw}{curveIndex}_{optimizerName}/full_result.npy', fullResult)
        messages = ["--------------------------------\n",
                    "Hardening parameters optimization completed",
                    "The full parameter solution is: ",
                    str(fullResult),
                    f"Succeeded iteration: {sim.fileNumber}",
                    "--------------------------------",
                    "Stage 5: CP Parameter Calibration completed",
                    "The final fitting parameter solution is: ",
                    str(fullResult)]

    pool = multiprocessing.Pool() # Creating a pool of multithreads that run jobs in parallel
    pool.map(parallelOptimization, curveIndices) # Obtain the optimized fitting parameters for the target curves in parallel
    pool.close()
    print("Fitting parameter optimization for all target curves completed")

if __name__ == '__main__':
    # External libraries
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    import multiprocessing
    from threading import Lock
    from modules.SIM import *
    from modules.preprocessing import *
    from modules.fitness import *
    from modules.helper import *
    #from optimizers.GA import *
    #from optimizers.BO import *
    #from optimizers.PSO import * 
    from os import path
    from optimize_config import *
    main()
# python optimize.py

