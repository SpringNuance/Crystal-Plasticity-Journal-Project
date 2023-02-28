# External libraries
import os
import numpy as np
from modules.SIM_damask2 import *
from initial_simulations import * 
from modules.preprocessing import *
from modules.stoploss import *
from modules.helper import *
from optimizers.GA import *
from optimizers.ANN import *
from prettytable import PrettyTable


def main():
    initial_processCurvesGlobal = np.load(f'results/{material}/{CPLaw}/universal/initial_processCurves.npy', allow_pickle=True).tolist()
    initial_trueCurvesGlobal = np.load(f'results/{material}/{CPLaw}/universal/initial_trueCurves.npy', allow_pickle=True).tolist()
    reverse_initial_trueCurvesGlobal = reverseAsParamsToLoading(initial_trueCurvesGlobal, loadings)
    reverse_initial_processCurvesGlobal = reverseAsParamsToLoading(initial_processCurvesGlobal, loadings)
    np.save(f"results/{material}/{CPLaw}/universal/reverse_initial_trueCurves.npy", reverse_initial_trueCurvesGlobal)
    np.save(f"results/{material}/{CPLaw}/universal/reverse_initial_processCurves.npy", reverse_initial_processCurvesGlobal)
    
    # Producing all target curves npy file
    getTargetCurves(material, CPLaw, curveIndex, loadings)

    print(f"Saving reverse and original initial true and process curves\n")
    print(f"Finished preparing all target curves\n\n")

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

        stage_CurvesList = np.load(f'{iterationPath}/stage_CurvesList.npy', allow_pickle=True).tolist()

        # Length of initial and iteration simulations
        initial_length = len(reverse_initial_processCurves)
        iteration_length = len(reverse_iteration_processCurves)
        
        messages.append(f"Curve {CPLaw}{curveIndex} status: \n")
        messages.append(f"{iteration_length} iteration simulations completed.\n")
        messages.append(f"{initial_length} initial simulations completed.\n")     
        messages.append(f"Total: {initial_length + iteration_length} simulations completed.\n\n")
        
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

        # Iteration curves info
        stage_CurvesList = []

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
        # DAMASK simulated curve used as experimental curve
        interpolatedStrain = interpolatingStrain(average_initialStrains[loading], exp_processCurve["strain"], exp_processCurve["stress"], yieldingPoints[CPLaw][loading], loading)                 
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
        for paramsTuple in combine_processCurves[loading]:
            sim_strain = combine_processCurves[loading][paramsTuple]["strain"]
            sim_stress = combine_processCurves[loading][paramsTuple]["stress"]
            combine_interpolateCurves[loading][paramsTuple] = {}
            combine_interpolateCurves[loading][paramsTuple]["strain"] = exp_curves["interpolate"][loading]["strain"] 
            combine_interpolateCurves[loading][paramsTuple]["stress"] = interpolatingStress(sim_strain, sim_stress, exp_curves["interpolate"][loading]["strain"], loading).reshape(-1) 

    reverse_combine_interpolateCurves = reverseAsParamsToLoading(combine_interpolateCurves, loadings)

    #np.save(f"{iterationPath}/interpolateCurves.npy", interpolateCurves)
    #np.save(f"{iterationPath}/reverse_interpolateCurves.npy", reverse_interpolateCurves)

    messages.append(f"Saving interpolating {CPLaw}{curveIndex} target curves\n")
    messages.append("Saving initial interpolating curves and reverse initial interpolating curves\n")
    messages.append("Experimental and simulated curves preparation completed\n\n")

    stringMessage = f"Curve {CPLaw}{curveIndex} info: \n"

    logTable = PrettyTable()

    logTable.field_names = ["Loading", "Exp σ_yield ", "Allowed σ_yield sim range"]
    for loading in loadings:
        targetYieldStress = '{:.3f}'.format(round(exp_curves["interpolate"][loading]['stress'][1], 3))
        rangeSimYieldBelow = '{:.3f}'.format(round(exp_curves["interpolate"][loading]["stress"][1] * (1 - linearYieldingDevGlobal * 0.01), 3))  
        rangeSimYieldAbove = '{:.3f}'.format(round(exp_curves["interpolate"][loading]['stress'][1] * (1 + linearYieldingDevGlobal * 0.01), 3))
        logTable.add_row([loading, f"{targetYieldStress} MPa", f"[{rangeSimYieldBelow}, {rangeSimYieldAbove} MPa]"])
    
    stringMessage += logTable.get_string()
    stringMessage += "\n\n"
    messages.append(stringMessage)
    if not os.path.exists(f"{iterationPath}/default_curves.npy"):
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
    else:
        default_curves = np.load(f"{iterationPath}/default_curves.npy", allow_pickle=True).tolist()
        default_params = default_curves["parameters_tuple"]
    
    if os.path.exists(f"{iterationPath}/default_curves.npy"):
        messages.append("The file default_curves.npy exists. Loading the default curves\n")
    
    stringMessage = f"Parameter set of the closest simulation curve to the target curve {CPLaw}{curveIndex} is: \n"
    
    logTable = PrettyTable()

    logTable.field_names = ["Parameter", "Value"]
    # logTable.align["Value"] = "l"

    for paramValue in default_params:
        exponent = param_infos[curveIndex][paramValue[0]]['exponent'] if param_infos[curveIndex][paramValue[0]]['exponent'] != "e0" else ""
        unit = paramsUnit[CPLaw][paramValue[0]]
        paramString = f"{paramValue[1]}"
        if exponent != "":
            paramString += exponent
        if unit != "":
            paramString += f" {unit}"
        logTable.add_row([paramValue[0], paramString])

    stringMessage += logTable.get_string()
    stringMessage += "\n"
    messages.append(stringMessage)
    if not os.path.exists(f"{iterationPath}/default_curves.npy"):
        messages.append("Saving the default curves\n")
    
    messages.append(f"This parameter set will serve as default parameters for the 4 stage optimization of curve {CPLaw}{curveIndex}\n\n")
    
    printList(messages)
