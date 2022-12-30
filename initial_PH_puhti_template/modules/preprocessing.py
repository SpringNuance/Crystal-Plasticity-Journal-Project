import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 
from modules.helper import * 
import time

#######################################################################
# Obtain and process parameter range, step size and rounding decimals # 
#######################################################################

def getParamRanges(material, CPLaw, curveIndices, searchingSpace, searchingType, roundContinuousDecimals, numOfColumns, startingColumn, spacing):
    general_range_df = pd.read_excel(f"targets/{material}/{CPLaw}/param_info/param_info.xlsx", usecols=[*range(1,7)], skiprows=1, engine="openpyxl")
    general_param_range = general_range_df.set_index(f'parameter').T.to_dict()
    for key in general_param_range:
        rangeOfParam = general_param_range[key]["general_range"].split("-")
        stepParam = general_param_range[key]["general_step"]
        rangeLow = rangeOfParam[0]
        rangeHigh = rangeOfParam[1]
        valueLow = float(rangeLow[1:])
        valueHigh = float(rangeHigh[:-1])
        boundaryLow = rangeLow[0]
        boundaryHigh = rangeHigh[-1]
        if searchingSpace == "discrete":
            if boundaryLow == "[":
                low = valueLow
            elif boundaryLow == "(":
                low = valueLow + stepParam
            if boundaryHigh == "]":
                high = valueHigh
            elif boundaryHigh == ")":
                low = valueHigh - stepParam
        elif searchingSpace == "continuous":
            if boundaryLow == "[":
                low = valueLow
            elif boundaryLow == "(":
                low = valueLow + 10 ** - roundContinuousDecimals
            if boundaryHigh == "]":
                high = valueHigh
            elif boundaryHigh == ")":
                low = valueHigh - 10 ** roundContinuousDecimals
        general_param_range[key].pop("general_range")
        general_param_range[key].pop("general_step")
        if general_param_range[key]["optimized_target"] == "yes":
            general_param_range[key]["optimized_target"] = True
        elif general_param_range[key]["optimized_target"] == "no":
            general_param_range[key]["optimized_target"] = False
        general_param_range[key]["step"] = stepParam
        general_param_range[key]["low"] = low
        general_param_range[key]["high"] = high
        frac, whole = math.modf(stepParam)
        roundingDecimal = 0
        if frac == 0:
            roundingDecimal = 0
        else:
            while whole == 0:
                stepParam *= 10
                roundingDecimal += 1
                frac, whole = math.modf(stepParam)
        general_param_range[key]["round"] = roundingDecimal
    #print(general_param_range)
    #time.sleep(60)
    np.save(f'targets/{material}/{CPLaw}/param_info/general_params.npy', general_param_range)
    
    if searchingType == "general":
        for index in curveIndices:
            np.save(f'targets/{material}/{CPLaw}/param_info/{CPLaw}{index}_params.npy', general_param_range)
    
    elif searchingType == "specific":
        for index in curveIndices:   
            indexMin1 = index - 1
            suffixMin1 = f".{indexMin1}"     
            suffix = f".{index}"
            beginningColumn = startingColumn + (index - 1) * spacing + (index - 1) * numOfColumns
            endingColumn = startingColumn + (index - 1) * spacing + index * numOfColumns
            df = pd.read_excel(f"targets/{material}/{CPLaw}/param_info/param_info.xlsx", usecols=[*range(beginningColumn, endingColumn)], skiprows=1, engine="openpyxl")
            df.rename(columns={f"parameter{suffix}": "parameter", f"range{suffixMin1}": "range", f"step{suffixMin1}": "step"}, inplace=True)
            #print(df)

            param_range_specific = df.set_index(f'parameter').T.to_dict()
            #print(param_range_specific)
            param_range = copy.deepcopy(general_param_range)

            for key in param_range:
                param_range[key]["range"] = param_range_specific[key]["range"]
                param_range[key]["step"] = param_range_specific[key]["step"]
                rangeOfParam = param_range[key]["range"].split("-")
                stepParam = param_range[key]["step"]
                rangeLow = rangeOfParam[0]
                rangeHigh = rangeOfParam[1]
                valueLow = float(rangeLow[1:])
                valueHigh = float(rangeHigh[:-1])
                boundaryLow = rangeLow[0]
                boundaryHigh = rangeHigh[-1]
                if searchingSpace == "discrete":
                    if boundaryLow == "[":
                        low = valueLow
                    elif boundaryLow == "(":
                        low = valueLow + stepParam
                    if boundaryHigh == "]":
                        high = valueHigh
                    elif boundaryHigh == ")":
                        low = valueHigh - stepParam
                elif searchingSpace == "continuous":
                    if boundaryLow == "[":
                        low = valueLow
                    elif boundaryLow == "(":
                        low = valueLow + 10 ** - roundContinuousDecimals
                    if boundaryHigh == "]":
                        high = valueHigh
                    elif boundaryHigh == ")":
                        low = valueHigh - 10 ** roundContinuousDecimals
                param_range[key].pop("range")
                param_range[key]["low"] = low
                param_range[key]["high"] = high
                frac, whole = math.modf(stepParam)
                roundingDecimal = 0
                if frac == 0:
                    roundingDecimal = 0
                else:
                    while whole == 0:
                        stepParam *= 10
                        roundingDecimal += 1
                        frac, whole = math.modf(stepParam)
                param_range[key]["round"] = roundingDecimal
            #print(param_range)
            np.save(f'targets/{material}/{CPLaw}/param_info/{CPLaw}{index}_params.npy', param_range)

def loadParamInfos(material, CPLaw, curveIndices):
    param_ranges = {}
    for index in curveIndices:
        param_range = np.load(f'targets/{material}/{CPLaw}/param_info/{CPLaw}{index}_params.npy', allow_pickle=True)
        param_ranges[index] = param_range.tolist()
    return param_ranges

def loadGeneralParam(material, CPLaw):
    general_param_range = np.load(f'targets/{material}/{CPLaw}/param_info/general_params.npy', allow_pickle=True)
    return general_param_range.tolist()

############################
# Obtain the target curves # 
############################

def getTargetCurves(material, CPLaw, curveIndices, loadings, stressUnit, convertUnit, numOfColumns, startingColumn, spacing, skiprows):
    for loading in loadings:    
        for index in curveIndices:   
            if index == 1:
                suffix = ""
            else: 
                suffix = ".{}".format(index - 1)     
            #true_df.to_csv(f'{loading}_targets/{material}/{CPLaw}/{CPLaw}{index}_true.csv', index=False)
            if loading == "linear_uniaxial_RD":
                beginningColumn = startingColumn + (index - 1) * spacing + (index - 1) * numOfColumns
                endingColumn = startingColumn + (index - 1) * spacing + index * numOfColumns
                true_df = pd.read_excel(f"{loading}_targets/{material}/{CPLaw}/target{CPLaw}.xlsx", usecols=[*range(beginningColumn, endingColumn)], skiprows=skiprows, engine="openpyxl")
                # print(true_df)
                true_df.rename(columns={f"True stress, {stressUnit}{suffix}": f"True stress, {stressUnit}", f"True strain, -{suffix}": "True strain, -"}, inplace=True)
                # print(true_df)
                true_df[f"True stress, {stressUnit}"] = true_df[f"True stress, {stressUnit}"] * convertUnit
                preprocessedCurves = preprocessLinear(trueStrain, trueStress)
            else:
                preprocessedCurves = preprocessNonlinear(trueStrain, trueStress, strainPathX, strainPathY, strainPathZ)
            flow_df.to_csv(f'targets/{material}/{CPLaw}/{CPLaw}{index}_process.csv', index=False)

###################################
# Preprocessing nonlinear loading #
###################################

def preprocessNonlinear(trueStrain, trueStress, strainPathX, strainPathY, strainPathZ):
    strainPathXprocess = strainPathX.copy()
    strainPathYprocess = strainPathY.copy()
    strainPathZprocess = strainPathZ.copy()
    turningIndices = turningStressPoints(trueStress)
    #print(turningIndices)
    #unloadingIndex = turningIndices[0]
    reloadingIndex = turningIndices[1]
    for i in range(reloadingIndex, trueStrain.size):
        strainPathXprocess[i] -= strainPathX[reloadingIndex]
        strainPathYprocess[i] -= strainPathY[reloadingIndex]
        strainPathZprocess[i] -= strainPathZ[reloadingIndex]
    strainReloading = (2/3 * (strainPathXprocess ** 2 + strainPathYprocess ** 2 + strainPathZprocess ** 2)) ** (1/2) + trueStrain[reloadingIndex]
    actualStrain = trueStrain.copy()
    for i in range(reloadingIndex, trueStrain.size):
        actualStrain[i] = strainReloading[i]
    return {"strain": actualStrain, "stress": trueStress}

def turningStressPoints(trueStress):
    differences = np.diff(trueStress)
    index = 1
    turningIndices = []
    while index < differences.size:
        if (differences[index - 1] <= 0 and differences[index] >= 0) or (differences[index - 1] >= 0 and differences[index] <= 0):
            turningIndices.append(index)
        index += 1
    return turningIndices

def preprocessDAMASKNonlinear(path):
    df = pd.read_csv(path, skiprows = 6, delimiter = "\t")
    trueStrain = df["Mises(ln(V))"].to_numpy().reshape(-1)
    trueStress = df["Mises(Cauchy)"].to_numpy().reshape(-1)
    strainPathX = df["1_ln(V)"].to_numpy().reshape(-1)
    strainPathY = df["5_ln(V)"].to_numpy().reshape(-1)
    strainPathZ = df["9_ln(V)"].to_numpy().reshape(-1)
    return preprocessNonlinear(trueStrain, trueStress, strainPathX, strainPathY, strainPathZ)

################################
# Preprocessing linear loading #
################################

def preprocessLinear(trueStrain, trueStress):
    # truePlasticStrain = trueStrain - trueElasticstrain = trueStrain - trueStress/Young's modulus
    Young = (trueStress[1] - trueStress[0]) / (trueStrain[1] - trueStrain[0])
    truePlasticStrain = trueStrain - trueStress / Young    
    return {"strain": truePlasticStrain, "stress": trueStress}

def preprocessDAMASKLinear(path):
    df = pd.read_csv(path, skiprows = 6, delimiter = "\t")
    trueStrain = df["Mises(ln(V))"].to_numpy().reshape(-1)
    trueStress = df["Mises(Cauchy)"].to_numpy().reshape(-1)
    return preprocessLinear(trueStrain, trueStress)   

##############################
# Obtain the original curves #
##############################

def preprocessDAMASKTrue(path):
    df = pd.read_csv(path, skiprows = 6, delimiter = "\t")
    trueStrain = df["Mises(ln(V))"].to_numpy()
    trueStress = df["Mises(Cauchy)"].to_numpy()
    return {"strain": trueStrain, "stress": trueStress}


