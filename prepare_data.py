# External libraries
import os
import numpy as np
import optimize_config
from modules.SIM_damask2 import *
from modules.preprocessing import *
from modules.helper import *
from prettytable import PrettyTable
    # -------------------------------------------------------------------
    #   Step 1: Loading progress and preparing data
    # -------------------------------------------------------------------

def main_prepare(info):

    server = info['server']
    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
    initialSims = info['initialSims']
    curveIndex = info['curveIndex']
    projectPath = info['projectPath']
    optimizerName = info['optimizerName']
    param_info = info['param_info']
    material = info['material']
    method = info['method']
    searchingSpace = info['searchingSpace']
    roundContinuousDecimals = info['roundContinuousDecimals']
    loadings = info['loadings']
    exampleLoading = info['exampleLoading']
    yieldingPoints = info['yieldingPoints']
    weightsYielding = info['weightsYielding']
    weightsHardening = info['weightsHardening']
    weightsLoading = info['weightsLoading']
    paramsFormatted = info['paramsFormatted']
    paramsUnit = info['paramsUnit']
    linearYieldingDev = info['linearYieldingDev']

    print(70 * "*" + "\n")
    print(f"Step 1: Loading progress and preparing data for curve {CPLaw}{curveIndex}\n\n")

    iterationResultPath = f"results/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}"
    initialResultPath = f"results/{material}/{CPLaw}/universal"

    # Extracting the initial simulation data
    
    initial_loadings_processCurves = {}
    initial_loadings_trueCurves = {}

    for loading in loadings:
        initial_loadings_trueCurves[loading] = np.load(f'results/{material}/{CPLaw}/universal/{loading}/initial_processCurves.npy', allow_pickle=True).tolist()
        initial_loadings_processCurves[loading] = np.load(f'results/{material}/{CPLaw}/universal/{loading}/initial_trueCurves.npy', allow_pickle=True).tolist()
    # print(list(initial_loadings_trueCurves[loading].keys())[0])
    # time.sleep(30)
    # Calculating average strain from initial simulations 
    average_initialStrains = {}
    for loading in loadings:
        average_initialStrains[loading] = np.array(list(map(lambda strainstress: strainstress["strain"], initial_loadings_processCurves[loading].values()))).mean(axis=0)
    
    # Producing all target curves npy file
    getTargetCurves(material, CPLaw, curveIndex, loadings)

    # Preparing the experimental curve of all loadings
    exp_curve = {}
    exp_curve["true"] = {}
    exp_curve["process"] = {}
    exp_curve["interpolate"] = {}

    # Loading the target curve, calculating the interpolating curve and save the compact data of target curve
    for loading in loadings:
        example_sim_stress = list(initial_loadings_processCurves[loading].values())[0]["stress"]
        exp_trueCurve = np.load(f'targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_true.npy', allow_pickle=True).tolist()
        exp_processCurve = np.load(f'targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_process.npy', allow_pickle=True).tolist()
        interpolatedStrain = interpolatingStrain(average_initialStrains[loading], exp_processCurve["strain"], example_sim_stress, yieldingPoints[CPLaw][loading], loading)                 
        interpolatedStress = interpolatingStress(exp_processCurve["strain"], exp_processCurve["stress"], interpolatedStrain, loading).reshape(-1)
        exp_interpolateCurve = {
            "strain": interpolatedStrain,
            "stress": interpolatedStress
        }
        exp_curve["true"][loading] = exp_trueCurve
        exp_curve["process"][loading] = exp_processCurve
        exp_curve["interpolate"][loading] = exp_interpolateCurve 
        print(exp_curve["process"])
        time.sleep(30)
        np.save(f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_interpolate.npy", exp_curve["interpolate"][loading])
    
    np.save(f"targets/{material}/{CPLaw}/{CPLaw}{curveIndex}_curves.npy", exp_curve)
    print(f"Finished preparing all target curves\n\n")
    ##################################################################

  

    # Loading iteration curves
    iteration_loadings_trueCurves = {}
    iteration_loadings_processCurves = {}

    if os.path.exists(f"{iterationResultPath}/{exampleLoading}/iteration_processCurves.npy"):
        for loading in loadings:
            # Loading iteration curves
            iteration_loadings_trueCurves[loading] = np.load(f'{iterationResultPath}/{loading}/iteration_trueCurves.npy', allow_pickle=True).tolist()
            iteration_loadings_processCurves[loading] = np.load(f'{iterationResultPath}/{loading}/iteration_processCurves.npy', allow_pickle=True).tolist()
        # Iteration curves info
        stage_CurvesList = np.load(f'{iterationResultPath}/stage_CurvesList.npy', allow_pickle=True).tolist()
    else:
        for loading in loadings:
            iteration_loadings_trueCurves[loading] = {}
            iteration_loadings_processCurves[loading] = {}
        # Iteration curves info
        stage_CurvesList = []

    # Create combined curves
    combined_loadings_trueCurves = {}
    combined_loadings_processCurves = {}
    
    # Updating the combine curves with the initial simulations and iteration curves 
    
    for loading in loadings:
        combined_loadings_trueCurves[loading] = {}
        combined_loadings_processCurves[loading] = {}
        
        combined_loadings_trueCurves[loading].update(initial_loadings_trueCurves[loading])
        combined_loadings_processCurves[loading].update(initial_loadings_processCurves[loading])

        combined_loadings_trueCurves[loading].update(iteration_loadings_trueCurves[loading])
        combined_loadings_processCurves[loading].update(iteration_loadings_processCurves[loading])
    

    initial_loadings_interpolateCurves = {}
    iteration_loadings_interpolateCurves = {}

    # Calculating the interpolated curves from combine curves and derive reverse_interpolate curves
 
    for loading in loadings:
        initial_loadings_interpolateCurves[loading] = {}
        for paramsTuple in initial_loadings_processCurves[loading]:
            sim_strain = initial_loadings_processCurves[loading][paramsTuple]["strain"]
            sim_stress = initial_loadings_processCurves[loading][paramsTuple]["stress"]
            initial_loadings_interpolateCurves[loading][paramsTuple] = {}
            initial_loadings_interpolateCurves[loading][paramsTuple]["strain"] = exp_curve["interpolate"][loading]["strain"] 
            initial_loadings_interpolateCurves[loading][paramsTuple]["stress"] = interpolatingStress(sim_strain, sim_stress, exp_curve["interpolate"][loading]["strain"], loading).reshape(-1) 

    for loading in loadings:
        iteration_loadings_interpolateCurves[loading] = {}
        for paramsTuple in iteration_loadings_processCurves[loading]:
            sim_strain = iteration_loadings_processCurves[loading][paramsTuple]["strain"]
            sim_stress = iteration_loadings_processCurves[loading][paramsTuple]["stress"]
            iteration_loadings_interpolateCurves[loading][paramsTuple] = {}
            iteration_loadings_interpolateCurves[loading][paramsTuple]["strain"] = exp_curve["interpolate"][loading]["strain"] 
            iteration_loadings_interpolateCurves[loading][paramsTuple]["stress"] = interpolatingStress(sim_strain, sim_stress, exp_curve["interpolate"][loading]["strain"], loading).reshape(-1) 
    
    
    
    # Updating the combine curves with the initial simulations and iteration curves 
    combined_loadings_interpolateCurves = {}
    for loading in loadings:
        combined_loadings_interpolateCurves[loading] = {}
        combined_loadings_interpolateCurves[loading].update(initial_loadings_interpolateCurves[loading])
        combined_loadings_interpolateCurves[loading].update(iteration_loadings_interpolateCurves[loading])

    reverse_initial_loadings_trueCurves = reverseAsParamsToLoading(initial_loadings_trueCurves, loadings)
    reverse_initial_loadings_processCurves = reverseAsParamsToLoading(initial_loadings_processCurves, loadings)
    reverse_initial_loadings_interpolateCurves = reverseAsParamsToLoading(initial_loadings_interpolateCurves, loadings)

    reverse_iteration_loadings_trueCurves = reverseAsParamsToLoading(iteration_loadings_trueCurves, loadings)
    reverse_iteration_loadings_processCurves = reverseAsParamsToLoading(iteration_loadings_processCurves, loadings)
    reverse_iteration_loadings_interpolateCurves = reverseAsParamsToLoading(iteration_loadings_interpolateCurves, loadings)

    reverse_combined_loadings_trueCurves = reverseAsParamsToLoading(combined_loadings_trueCurves, loadings)
    reverse_combined_loadings_processCurves = reverseAsParamsToLoading(combined_loadings_processCurves, loadings)
    reverse_combined_loadings_interpolateCurves = reverseAsParamsToLoading(combined_loadings_interpolateCurves, loadings)

    ##########################

    stringMessage = f"Curve {CPLaw}{curveIndex} info: \n"

    logTable = PrettyTable()

    logTable.field_names = ["Loading", "Exp σ_yield ", "Allowed σ_yield sim range"]
    for loading in loadings:
        if loading.startswith("linear"):
            targetYieldStress = '{:.3f}'.format(round(exp_curve["interpolate"][loading]['stress'][1], 3))
            rangeSimYieldBelow = '{:.3f}'.format(round(exp_curve["interpolate"][loading]["stress"][1] * (1 - linearYieldingDev * 0.01), 3))  
            rangeSimYieldAbove = '{:.3f}'.format(round(exp_curve["interpolate"][loading]['stress'][1] * (1 + linearYieldingDev * 0.01), 3))
            logTable.add_row([loading, f"{targetYieldStress} MPa", f"[{rangeSimYieldBelow}, {rangeSimYieldAbove} MPa]"])
    
    stringMessage += logTable.get_string()
    stringMessage += "\n\n"
    print(stringMessage)

    if not os.path.exists(f"{iterationResultPath}/common/default_curves.npy"):
        tupleParamsStresses = list(reverse_initial_loadings_interpolateCurves.items())
        #print(tupleParamsStresses[0])
        #print(exp_curve["interpolate"])
        sortedClosestHardening = list(sorted(tupleParamsStresses, key = lambda paramsStresses: fitnessHardeningAllLoadings(exp_curve["interpolate"], paramsStresses[1], loadings, weightsLoading, weightsHardening)))

        # Obtaining the default hardening parameters
        default_params = sortedClosestHardening[0][0]

        default_curves = {}
        default_curves["iteration"] = 0
        default_curves["stage"] = 0
        default_curves["parameters_tuple"] = default_params
        default_curves["parameters_dict"] = dict(default_params)
        default_curves["true"] = reverse_initial_loadings_trueCurves[default_params]
        default_curves["process"] = reverse_initial_loadings_processCurves[default_params]
        default_curves["interpolate"] = reverse_initial_loadings_interpolateCurves[default_params]
        default_curves["yielding_loss"] = calculateMSE(exp_curve["interpolate"], default_curves["interpolate"], "yielding", loadings, weightsLoading, weightsYielding, weightsHardening)
        default_curves["hardening_loss"] = calculateMSE(exp_curve["interpolate"], default_curves["interpolate"], "hardening", loadings, weightsLoading, weightsYielding, weightsHardening)
        np.save(f"{iterationResultPath}/common/default_curves.npy", default_curves)
        print("Saving the default curves\n")
    else:
        default_curves = np.load(f"{iterationResultPath}/common/default_curves.npy", allow_pickle=True).tolist()
        print("The file default_curves.npy exists. Loading the default curves\n")

    # Length of initial and iteration simulations
    initial_length = len(reverse_initial_loadings_processCurves)
    iteration_length = len(reverse_iteration_loadings_processCurves)

    print(f"Curve {CPLaw}{curveIndex} status: \n")
    print(f"{iteration_length} iteration simulations completed.\n")
    print(f"{initial_length} initial simulations completed.\n")     
    print(f"Total: {initial_length + iteration_length} simulations completed.\n")
    print(f"Experimental and simulated curves preparation for {CPLaw}{curveIndex} has completed\n")

    print(f"Parameter set of the closest simulation curve to the target curve {CPLaw}{curveIndex} is: \n")
    
    printParametersClean(default_curves["parameters_tuple"], param_info, paramsUnit, CPLaw)

    print(f"The default parameters for the optimization of curve {CPLaw}{curveIndex}\n")
    time.sleep(30)
    np.save(f'{initialResultPath}/common/initial_loadings_trueCurves', initial_loadings_trueCurves) 
    np.save(f'{initialResultPath}/common/initial_loadings_processCurves', initial_loadings_processCurves)
    np.save(f'{initialResultPath}/common/initial_loadings_interpolateCurves', initial_loadings_interpolateCurves)
    np.save(f'{initialResultPath}/common/reverse_initial_loadings_trueCurves', reverse_initial_loadings_trueCurves)
    np.save(f'{initialResultPath}/common/reverse_initial_loadings_processCurves', reverse_initial_loadings_processCurves)
    np.save(f'{initialResultPath}/common/reverse_initial_loadings_interpolateCurves', reverse_initial_loadings_interpolateCurves)      
    
    np.save(f'{iterationResultPath}/common/iteration_loadings_trueCurves', iteration_loadings_trueCurves)
    np.save(f'{iterationResultPath}/common/iteration_loadings_processCurves', iteration_loadings_processCurves)
    np.save(f'{iterationResultPath}/common/iteration_loadings_interpolateCurves', iteration_loadings_interpolateCurves)
    np.save(f'{iterationResultPath}/common/reverse_iteration_loadings_trueCurves', reverse_iteration_loadings_trueCurves)
    np.save(f'{iterationResultPath}/common/reverse_iteration_loadings_processCurves', reverse_iteration_loadings_processCurves)
    np.save(f'{iterationResultPath}/common/reverse_iteration_loadings_interpolateCurves', reverse_iteration_loadings_interpolateCurves)
    
    np.save(f'{iterationResultPath}/common/combined_loadings_trueCurves', combined_loadings_trueCurves)
    np.save(f'{iterationResultPath}/common/combined_loadings_processCurves', combined_loadings_processCurves)
    np.save(f'{iterationResultPath}/common/combined_loadings_interpolateCurves', combined_loadings_interpolateCurves)
    np.save(f'{iterationResultPath}/common/reverse_combined_loadings_trueCurves', reverse_combined_loadings_trueCurves)
    np.save(f'{iterationResultPath}/common/reverse_combined_loadings_processCurves', reverse_combined_loadings_processCurves)
    np.save(f'{iterationResultPath}/common/reverse_combined_loadings_interpolateCurves', reverse_combined_loadings_interpolateCurves)
    
    prepared_data = {
        'initial_length': initial_length,
        'iteration_length': iteration_length,
        'exp_curve': exp_curve,
        'initialResultPath': initialResultPath,
        'iterationResultPath': iterationResultPath,
        'stage_CurvesList': stage_CurvesList,

        'initial_loadings_trueCurves': initial_loadings_trueCurves, 
        'initial_loadings_processCurves': initial_loadings_processCurves,
        'initial_loadings_interpolateCurves': initial_loadings_interpolateCurves,
        'reverse_initial_loadings_trueCurves': reverse_initial_loadings_trueCurves,
        'reverse_initial_loadings_processCurves': reverse_initial_loadings_processCurves,
        'reverse_initial_loadings_interpolateCurves': reverse_initial_loadings_interpolateCurves,        
        'iteration_loadings_trueCurves': iteration_loadings_trueCurves, 
        'iteration_loadings_processCurves': iteration_loadings_processCurves,
        'iteration_loadings_interpolateCurves': iteration_loadings_interpolateCurves,
        'reverse_iteration_loadings_trueCurves': reverse_iteration_loadings_trueCurves,
        'reverse_iteration_loadings_processCurves': reverse_iteration_loadings_processCurves,
        'reverse_iteration_loadings_interpolateCurves': reverse_iteration_loadings_interpolateCurves,
        'combined_loadings_trueCurves': combined_loadings_trueCurves,
        'combined_loadings_processCurves': combined_loadings_processCurves,
        'combined_loadings_interpolateCurves': combined_loadings_interpolateCurves,
        'reverse_combined_loadings_trueCurves': reverse_combined_loadings_trueCurves,
        'reverse_combined_loadings_processCurves': reverse_combined_loadings_processCurves,
        'reverse_combined_loadings_interpolateCurves': reverse_combined_loadings_interpolateCurves,
    }
    time.sleep(60)
    return prepared_data

if __name__ == '__main__':
    info = optimize_config.main()
    main_prepare(info)