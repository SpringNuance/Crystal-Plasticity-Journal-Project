# External libraries
import os
import numpy as np
import optimize_config
from modules.SIM_damask2 import *
from modules.preprocessing import *
from modules.helper import *
from prettytable import PrettyTable

def main_stagesAnalysis(info):    
    messages = []
    messages.append(70 * "*" + "\n")
    messages.append(f"Step 3: Assessment of number optimization stages and level of deviation percentage of curve {CPLaw}{curveIndex}\n\n")

    allParams = dict(default_params).keys()

    yieldingParams = list(filter(lambda param: param_info[param]["type"] == "yielding", allParams))
    large_hardeningParams = list(filter(lambda param: param_info[param]["type"] == "large_hardening", allParams))
    small_hardeningParams = list(filter(lambda param: param_info[param]["type"] == "small_hardening", allParams))
    
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
        messages.append("There are no non-Bauschinger effect hardening parameters\n")
        messages.append("2nd stage optimization not required\n")
    else:
        messages.append(f"There are {len(large_hardeningParams)} large hardening parameters\n")
        messages.append("2nd stage optimization required\n")

    if len(small_hardeningParams) == 0:
        messages.append("There are no Bauschinger effect hardening parameters\n")
        messages.append("3rd stage optimization not required\n\n")
    else:
        messages.append(f"There are {len(small_hardeningParams)} small hardening parameters\n")
        messages.append("3rd stage optimization required\n\n")

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
    firstHardeningDevs = {}
    secondHardeningDevs = {}
    for loading in loadings:
        if loading.startswith("linear"):
            yieldingDevs[loading] = linearYieldingDev
            firstHardeningDevs[loading] = firstLinearHardeningDev
            secondHardeningDevs[loading] = secondLinearHardeningDev
        else:
            firstHardeningDevs[loading] = firstNonlinearHardeningDev
            secondHardeningDevs[loading] = secondNonlinearHardeningDev 
    # ----------------------------------------------------------------------------
    #   Four optimization stage: Optimize the parameters for the curves in parallel 
    # ----------------------------------------------------------------------------
    deviationPercent = [yieldingDevs, firstHardeningDevs, secondHardeningDevs]
    deviationCondition = [insideYieldingDevLinear, insideHardeningDevAllLoadings, insideHardeningDevAllLoadings]
    optimize_params = [yieldingParams, large_hardeningParams, small_hardeningParams]
    parameterType = ["yielding", "first hardening", "second hardening"]
    optimize_type = ["yielding", "hardening", "hardening"]
    ordinalUpper = ["First", "Second", "Third"]
    ordinalLower = ["first", "second", "third"]
    ordinalNumber = ["1","2","3"]

if __name__ == '__main__':
    info = optimize_config.main()
    main_stagesAnalysis(info)