import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math 
import copy
import time

#######################################################################
# Obtain and process parameter range, step size and rounding decimals # 
#######################################################################

def getParamRanges(material, CPLaw, curveIndex, searchingSpace, roundContinuousDecimals):
    general_range_df = pd.read_excel(f"targets/{material}/{CPLaw}/param_info/param_info.xlsx", engine="openpyxl")
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
    
    np.save(f'targets/{material}/{CPLaw}/param_info/{CPLaw}{curveIndex}_params.npy', general_param_range)
    
def loadParamInfos(material, CPLaw, curveIndex):
    param_ranges = {}
    param_range = np.load(f'targets/{material}/{CPLaw}/param_info/{CPLaw}{curveIndex}_params.npy', allow_pickle=True)
    param_ranges = param_range.tolist()
    return param_ranges

def loadGeneralParam(material, CPLaw):
    general_param_range = np.load(f'targets/{material}/{CPLaw}/param_info/general_params.npy', allow_pickle=True)
    return general_param_range.tolist()

############################
# Obtain the target curves # 
############################

def getTargetCurves(material, CPLaw, curveIndex, loadings, Pa=True):
    for loading in loadings:    
        #print(loading)
        path = f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}.xlsx"
        trueCurve = preprocessExperimentalTrue(path, Pa)
        processCurve = preprocessExperimentalFitted(path, Pa) 
        #print("True curve")
        #print(trueCurve)
        #print("Processed curve")
        #print(processCurve)
        np.save(f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_true.npy", trueCurve)
        np.save(f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_process.npy", processCurve)

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
    # Equivalent Von Mises strain formula
    strainReloading = (2/3 * (strainPathXprocess ** 2 + strainPathYprocess ** 2 + strainPathZprocess ** 2)) ** (1/2) + trueStrain[reloadingIndex]
    actualStrain = trueStrain.copy()
    for i in range(reloadingIndex, trueStrain.size):
        actualStrain[i] = strainReloading[i]
    return {"strain": actualStrain, "stress": trueStress}

def turningStressPoints(trueStress):
    #print(trueStress)
    differences = np.diff(trueStress)
    index = 1
    turningIndices = []
    while index < differences.size:
        if (differences[index - 1] <= 0 and differences[index] >= 0) or (differences[index - 1] >= 0 and differences[index] <= 0):
            turningIndices.append(index)
        index += 1
    return turningIndices

def preprocessDAMASKNonlinear(path, excel=False):
    if not excel:
        df = pd.read_csv(path, skiprows = 6, delimiter = "\t")
    else:
        df = pd.read_excel(path, usecols=["Mises(Cauchy)","Mises(ln(V))","1_ln(V)","5_ln(V)","9_ln(V)"], skiprows=6, engine="openpyxl")
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

def preprocessDAMASKLinear(path, excel=False):
    if not excel:
        df = pd.read_csv(path, skiprows = 6, delimiter = "\t")
    else:
        df = pd.read_excel(path, usecols=["Mises(Cauchy)","Mises(ln(V))"], skiprows=6, engine="openpyxl")
    trueStrain = df["Mises(ln(V))"].to_numpy().reshape(-1)
    trueStress = df["Mises(Cauchy)"].to_numpy().reshape(-1)
    return preprocessLinear(trueStrain, trueStress)   

##############################
# Obtain the original curves #
##############################

def preprocessDAMASKTrue(path, excel=False):
    if not excel:
        df = pd.read_csv(path, skiprows = 6, delimiter = "\t")
    else:
        df = pd.read_excel(path, usecols=["Mises(Cauchy)","Mises(ln(V))"], skiprows=6, engine="openpyxl")
    trueStrain = df["Mises(ln(V))"].to_numpy()
    trueStress = df["Mises(Cauchy)"].to_numpy()
    return {"strain": trueStrain, "stress": trueStress}

def preprocessExperimentalTrue(path, Pa):
    if Pa:
        df = pd.read_excel(path, usecols=["exp_strain","exp_stress (Pa)"], engine="openpyxl")
        trueStrain = df["exp_strain"].to_numpy()
        trueStress = df["exp_stress (Pa)"].to_numpy()
    else:
        df = pd.read_excel(path, usecols=["exp_strain","exp_stress (MPa)"], engine="openpyxl")
        trueStrain = df["exp_strain"].to_numpy()
        trueStress = df["exp_stress (MPa)"].to_numpy()
    trueStrain = trueStrain[~np.isnan(trueStrain)]
    trueStress = trueStress[~np.isnan(trueStress)]
    return {"strain": trueStrain, "stress": trueStress}

def preprocessExperimentalFitted(path, Pa):
    if Pa:
        df = pd.read_excel(path, usecols=["fitted_strain","fitted_stress (Pa)"], engine="openpyxl")
        fittedStrain = df["fitted_strain"].to_numpy()
        fittedStress = df["fitted_stress (Pa)"].to_numpy()
    else:
        df = pd.read_excel(path, usecols=["fitted_strain","fitted_stress (MPa)"], engine="openpyxl")
        fittedStrain = df["fitted_strain"].to_numpy()
        fittedStress = df["fitted_stress (MPa)"].to_numpy()
    fittedStrain = fittedStrain[~np.isnan(fittedStrain)]
    fittedStress = fittedStress[~np.isnan(fittedStress)]
    return {"strain": fittedStrain, "stress": fittedStress}

######################################################################
# Generalized Voce fitting equation [Ureta Xavier] #
######################################################################

# According to Ureta
# tau0 = 127.2 MPa
# tau1 = 124.2 MPa
# theta1 = 203.5 MPa
# theta0/tau1 = 17.74 => theta0 = 2203 MPa
# y = tau0 + (tau1 + theta1 * x) * (1 - exp(- x * abs(theta0/tau1)))

def preprocessSwiftVoceHardening(trueStrain, tau0, tau1, theta0, theta1):
    trueStress = tau0 + (tau1 + theta1 * trueStrain) * (1 - math.exp(- trueStrain * abs(theta0/tau1)))
    return {"strain": trueStrain, "stress": trueStress}


###################################
# Calculate interpolating strains #
###################################

def getIndexBeforeStrainLevel(strain, level):
    for i in range(len(strain)):
        if strain[i] > level:
            return i - 1

def getIndexAfterStrainLevel(strain, level):
    for i in range(len(strain)):
        if strain[i] > level:
            return i

def interpolatingStrain(average_initialStrain, exp_strain, sim_stress, yieldingPoint, loading):
    if loading.startswith("linear"):
        beforeYieldingIndex = getIndexBeforeStrainLevel(average_initialStrain, yieldingPoint) 
        interpolatedStrain = average_initialStrain[beforeYieldingIndex:]
        # Strain level is added to the interpolating strains
        interpolatedStrain = np.insert(interpolatedStrain, 1, yieldingPoint)   
        if interpolatedStrain[-1] > exp_strain[-1]:
            indexOfInterpolatedStrainAfterLastExpStrain = getIndexAfterStrainLevel(interpolatedStrain, exp_strain[-1])
            interpolatedStrain = interpolatedStrain[:indexOfInterpolatedStrainAfterLastExpStrain+1]
    else: 
        reloadingIndex = turningStressPoints(sim_stress)[1]
        interpolatedStrain = average_initialStrain[reloadingIndex:]
        beforeYieldingIndex = getIndexBeforeStrainLevel(interpolatedStrain, yieldingPoint)
        interpolatedStrain = interpolatedStrain[beforeYieldingIndex:]
        interpolatedStrain = np.insert(interpolatedStrain, 1, yieldingPoint)
        if interpolatedStrain[-1] > exp_strain[-1]:
            indexOfInterpolatedStrainAfterLastExpStrain = getIndexAfterStrainLevel(interpolatedStrain, exp_strain[-1])
            interpolatedStrain = interpolatedStrain[:indexOfInterpolatedStrainAfterLastExpStrain+1]
    #print(interpolatedStrain)
    return interpolatedStrain 

def interpolatingStress(strain, stress, interpolatedStrain, loading):
    # interpolated function fits the stress-strain curve data 
    #print(stress)
    if loading.startswith("linear"):
        # Allows extrapolation
        interpolatingFunction = interp1d(strain, stress, fill_value='extrapolate')
        # Calculate the stress values at the interpolated strain points
        interpolatedStress = interpolatingFunction(interpolatedStrain).reshape(-1)
        
    else:
        if len(turningStressPoints(stress)) != 0:
            reloadingIndex = turningStressPoints(stress)[1]
            strain = strain[reloadingIndex:]
            stress = stress[reloadingIndex:]
        # Allows extrapolation
        interpolatingFunction = interp1d(strain, stress, fill_value='extrapolate')
        # Calculate the stress values at the interpolated strain points
        interpolatedStress = interpolatingFunction(interpolatedStrain).reshape(-1)
    #print(interpolatedStress)
    return interpolatedStress 

def multiplyStressByUnit(curves, convertUnit):
    rearrangeCurves = {}

    # create a new dictionary with stress multiplied by 1e6

    for key, value in curves.items():
        new_value = copy.deepcopy(value)
        new_value['stress'] = value['stress'] * convertUnit
        rearrangeCurves[key] = new_value
    
    return rearrangeCurves