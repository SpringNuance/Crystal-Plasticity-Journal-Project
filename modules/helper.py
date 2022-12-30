from audioop import reverse
from math import *
import copy
import numpy as np
from modules.stoploss import *


from modules.helper import *
import sys, os
import random
import time

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

def round_to_step(low, step, value, roundDecimals):
    upperBound = floor((value - low)/step) * step + low
    lowerBound = ceil((value - low)/step) *step + low
    upperDif = upperBound - value
    lowerDif = value - lowerBound
    if upperDif >= lowerDif:
        return round(upperBound, roundDecimals)
    else: 
        return round(lowerBound, roundDecimals)

# params and param_info are both dictionaries
def round_discrete(params, param_info):
    for parameter in params:
        params[parameter] = round_to_step(param_info[parameter]['low'], param_info[parameter]['step'], params[parameter], param_info[parameter]['round'])
    return params

def round_continuous(params, roundContinuousDecimals):
    for parameter in params:
        params[parameter] = round(params[parameter], roundContinuousDecimals)
    return params

def reverseAsParamsToLoading(curves, loadings):
    reverseCurves = {}
    for paramsTuple in curves['linear_uniaxial_RD']:
        reverseCurves[paramsTuple] = {}
        for loading in loadings:
            reverseCurves[paramsTuple][loading] = curves[loading][paramsTuple]
    return reverseCurves

##############
# Param info #
##############

def param_info_GA_discrete_func(param_info):
    temporary_param_info = {}
    for param in param_info:
        temporary_param_info[param] = {}
        temporary_param_info[param]['low'] = param_info[param]['low']
        temporary_param_info[param]['high'] = param_info[param]['high']
        temporary_param_info[param]['step'] = param_info[param]['step']
    return temporary_param_info 

def param_infos_GA_discrete_func(param_infos):
    temporary_param_infos = {}
    for curveIndex in param_infos:
        temporary_param_infos[curveIndex] = {}
        for param in param_infos[curveIndex]:
            temporary_param_infos[curveIndex][param] = {}
            temporary_param_infos[curveIndex][param]['low'] = param_infos[curveIndex][param]['low']
            temporary_param_infos[curveIndex][param]['high'] = param_infos[curveIndex][param]['high']
            temporary_param_infos[curveIndex][param]['step'] = param_infos[curveIndex][param]['step']
    return temporary_param_infos 

def param_info_GA_continuous_func(param_info):
    temporary_param_info = {}
    for param in param_info:
        temporary_param_info[param] = {}
        temporary_param_info[param]['low'] = param_info[param]['low']
        temporary_param_info[param]['high'] = param_info[param]['high']
    return temporary_param_info 

def param_infos_GA_continuous_func(param_infos):
    temporary_param_infos = {}
    for curveIndex in param_infos:
        temporary_param_infos[curveIndex] = {}
        for param in param_infos[curveIndex]:
            temporary_param_infos[curveIndex][param] = {}
            temporary_param_infos[curveIndex][param]['low'] = param_infos[curveIndex][param]['low']
            temporary_param_infos[curveIndex][param]['high'] = param_infos[curveIndex][param]['high']
    return temporary_param_infos 

def param_infos_PSO_low_func(param_infos):
    temporary_param_infos = {}
    for curveIndex in param_infos:
        temporary_param_infos[curveIndex] = {}
        for param in param_infos[curveIndex]:
            temporary_param_infos[curveIndex][param] = {}
            temporary_param_infos[curveIndex][param]['low'] = param_infos[curveIndex][param]['low']
    return temporary_param_infos 

def param_infos_PSO_high_func(param_infos):
    temporary_param_infos = {}
    for curveIndex in param_infos:
        temporary_param_infos[curveIndex] = {}
        for param in param_infos[curveIndex]:
            temporary_param_infos[curveIndex][param] = {}
            temporary_param_infos[curveIndex][param]['high'] = param_infos[curveIndex][param]['high']
    return temporary_param_infos 

def param_info_BO_func(param_info):
    temporary_param_info = {}
    for param in param_info:
        temporary_param_info[param] = (param_info[param]['low'], param_info[param]['high'])
    return temporary_param_info 

def param_infos_BO_func(param_infos):
    temporary_param_infos = {}
    for curveIndex in param_infos:
        temporary_param_infos[curveIndex] = {}
        for param in param_infos[curveIndex]:
            temporary_param_infos[curveIndex][param] = (param_infos[curveIndex][param]['low'], param_infos[curveIndex][param]['high'])
    return temporary_param_infos 

###############################################
# Calculate fitness functions in a dictionary #
###############################################

def calculateMSE(exp_interpolateCurves, sim_interpolateCurves, optimize_type, loadings, weightsLoading, weightsYielding, weightsHardening):
    MSE = {}
    if optimize_type == "yielding":
        MSE["weighted_total_MSE"] = fitnessYieldingAllLoadings(exp_interpolateCurves, sim_interpolateCurves, loadings, weightsLoading, weightsYielding)
        for loading in loadings:
            if loading == "linear_uniaxial_RD":
                MSE[loading] = {}
                MSE[loading]["Y1"] = Y1Linear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"])
                MSE[loading]["Y2"] = Y2Linear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"])
                MSE[loading]["weighted_loading_MSE"] = fitnessYieldingLinear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"], weightsYielding)
            else:
                MSE[loading] = {}
                MSE[loading]["Y1"] = Y1Nonlinear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"])
                MSE[loading]["Y2"] = Y2Nonlinear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"])
                MSE[loading]["weighted_loading_MSE"] = fitnessYieldingNonlinear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"], weightsYielding)
    
    if optimize_type == "hardening":
        MSE["weighted_total_MSE"] = fitnessHardeningAllLoadings(exp_interpolateCurves, sim_interpolateCurves, loadings, weightsLoading, weightsHardening)
        for loading in loadings:
            if loading == "linear_uniaxial_RD":
                MSE[loading] = {}
                MSE[loading]["H1"] = H1Linear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"])
                MSE[loading]["H2"] = H2Linear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"])
                MSE[loading]["weighted_loading_MSE"] = fitnessHardeningLinear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"], weightsHardening)
            else:
                MSE[loading] = {}
                MSE[loading]["H1"] = H1Nonlinear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"])
                MSE[loading]["H2"] = H2Nonlinear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"])
                MSE[loading]["weighted_loading_MSE"] = fitnessHardeningNonlinear(exp_interpolateCurves[loading]["stress"], sim_interpolateCurves[loading]["stress"], exp_interpolateCurves[loading]["strain"], weightsHardening)
    return MSE
