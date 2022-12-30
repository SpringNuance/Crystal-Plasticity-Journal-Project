from math import *
import copy
import numpy as np
from scipy.interpolate import interp1d
from modules.helper import *
import sys, os
import random

def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)
    
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def round_to_step(low, step, value, roundValue):
    upperBound = floor((value - low)/step) * step + low
    lowerBound = ceil((value - low)/step) *step + low
    upperDif = upperBound - value
    lowerDif = value - lowerBound
    if upperDif >= lowerDif:
        return round(upperBound, roundValue)
    else: 
        return round(lowerBound, roundValue)

# params and param_info are both dictionaries
def round_discrete(params, param_info):
    for parameter in params:
        params[parameter] = round_to_step(param_info[parameter]['low'], param_info[parameter]['step'], params[parameter], param_info[parameter]['round'])
    return params

def round_continuous(params, roundContinuousDecimals):
    for parameter in params:
        params[parameter] = round(params[parameter], roundContinuousDecimals)
    return params



# Requiring that the interpolatedStrain must lie inside the range of strain
def interpolatedStressFunction(stress, strain, interpolatedStrain):
    # interpolated function fits the stress-strain curve data 
    interpolatedFunction = interp1d(strain, stress)
    # Calculate the stress values at the interpolated strain points
    interpolatedStress = interpolatedFunction(interpolatedStrain)
    return interpolatedStress 

def param_info_no_round_func(param_info):
    param_info_no_round = {}
    temporary_param_info = copy.deepcopy(param_info)
    for key in param_info:
        temporary_param_info[key].pop('round')
        param_info_no_round[key] = temporary_param_info[key]
    return param_info_no_round

def param_infos_no_round_func(param_infos):
    param_infos_no_round = {}
    temporary_param_infos = copy.deepcopy(param_infos)
    for index in param_infos:
        param_infos_no_round[index] = {}
        for param in param_infos[index]:
            temporary_param_infos[index][param].pop('round')
            param_infos_no_round[index][param] = temporary_param_infos[index][param]
    return param_infos_no_round

def param_info_no_step_tuple_func(param_info_no_round):
    param_info_no_step = {}
    temporary_param_info = copy.deepcopy(param_info_no_round)
    for key in param_info_no_round:
        temporary_param_info[key].pop('step')
        param_info_no_step[key] = (temporary_param_info[key]['low'], temporary_param_info[key]['high'])
    return param_info_no_step

def param_infos_no_step_tuple_func(param_infos_no_round):
    param_infos_no_step = {}
    temporary_param_infos = copy.deepcopy(param_infos_no_round)
    for index in param_infos_no_round:
        param_infos_no_step[index] = {}
        for param in param_infos_no_round[index]:
            param_infos_no_step[index][param] = (temporary_param_infos[index][param]['low'], temporary_param_infos[index][param]['high'])
    return param_infos_no_step

def param_info_no_step_dict_func(param_info_no_round):
    param_info_no_step = {}
    temporary_param_info = copy.deepcopy(param_info_no_round)
    for key in param_info_no_round:
        temporary_param_info[key].pop('step')
        param_info_no_step[key] = temporary_param_info[key]
    return param_info_no_step

def param_infos_no_step_dict_func(param_infos_no_round):
    param_infos_no_step = {}
    temporary_param_infos = copy.deepcopy(param_infos_no_round)
    for index in param_infos_no_round:
        param_infos_no_step[index] = {}
        for param in param_infos_no_round[index]:
            temporary_param_infos[index][param].pop('step')
            param_infos_no_step[index][param] = temporary_param_infos[index][param]
    return param_infos_no_step

def rearrangePH(params):
    newParams = {}
    newParams['a'] = params['a']
    newParams['h0'] = params['h0']
    newParams['tau0'] = params['tau0']
    newParams['tausat'] = params['tausat']
    return newParams
    
def rearrangeDB(params):
    newParams = {}
    newParams['dipole'] = params['dipole']
    newParams['islip'] = params['islip']
    newParams['omega'] = params['omega']
    newParams['p'] = params['p']
    newParams['q'] = params['q']
    newParams['tausol'] = params['tausol']
    return newParams

def rearrangeParamRange(param_info):
    new_param_info = {}
    for key in param_info:
        new_param_info[key] = {}
        new_param_info[key]['low'] = param_info[key]['low']
        new_param_info[key]['high'] = param_info[key]['high']
        new_param_info[key]['step'] = param_info[key]['step']
        new_param_info[key]['round'] = param_info[key]['round']
    return new_param_info

def tupleOrListToDict(params, CPLaw):
    newParams = {}
    if CPLaw == "PH":
        newParams['a'] = params[0]
        newParams['h0'] = params[1]
        newParams['tau0'] = params[2]
        newParams['tausat'] = params[3]
        newParams["self"] = params[4]
        newParams["coplanar"] = params[5]
        newParams["collinear"] = params[6]
        newParams["orthogonal"] = params[7]
        newParams["glissile"] = params[8]
        newParams["sessile"] = params[9]
    if CPLaw == "DB":
        newParams['dipole'] = params[0]
        newParams['islip'] = params[1]
        newParams['omega'] = params[2]
        newParams['p'] = params[3]
        newParams['q'] = params[4]
        newParams['tausol'] = params[5]
        newParams["self"] = params[6]
        newParams["coplanar"] = params[7]
        newParams["collinear"] = params[8]
        newParams["orthogonal"] = params[9]
        newParams["glissile"] = params[10]
        newParams["sessile"] = params[11]
    return newParams

def defaultParams(partialResult, CPLaw, default_yield_value):
    if CPLaw == "PH":
        solution_dict = {
            'a': default_yield_value['a'],
            'h0': default_yield_value['h0'],
            'tau0': partialResult['tau0'],
            'tausat': default_yield_value['tausat']
        }
    elif CPLaw == "DB":
        solution_dict = {
            'dipole': default_yield_value['dipole'],
            'islip': default_yield_value['islip'],
            'omega': default_yield_value['omega'],
            'p': partialResult["p"],
            'q': partialResult["q"], 
            'tausol': partialResult["tausol"]
        }
    return solution_dict

def getIndexBeforeStrainLevel(strain, level):
    for i in range(len(strain)):
        if strain[i] > level:
            return i - 1

def getIndexBeforeStrainLevelEqual(strain, level):
    for i in range(len(strain)):
        if strain[i] >= level:
            return i

def calculateInterpolatingStrains(allStrains, limitingStrain, mainStrain, yieldStressStrainLevel):
    minStrain = min(list(map(lambda x: x[-1], allStrains)))
    # print(minStrain)
    upperMainIndex = getIndexBeforeStrainLevelEqual(mainStrain, minStrain)
    
    # print(mainStrain[upperMainIndex])
    mainStrain = mainStrain[:upperMainIndex]
    # print(upperMainIndex)
    # print(mainStrain)
    x_max = limitingStrain.max() 
    # print(limitingStrain)
    indexUpper = getIndexBeforeStrainLevelEqual(mainStrain, x_max)
    indexLower = getIndexBeforeStrainLevel(mainStrain, yieldStressStrainLevel) 
    mainStrain = mainStrain[:indexUpper]
    mainStrain = mainStrain[indexLower:]

    interpolatedStrain = mainStrain

    # Strain level is added to the interpolating strains
    interpolatedStrain = np.insert(interpolatedStrain, 1, yieldStressStrainLevel)
    return interpolatedStrain