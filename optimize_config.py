from lib2to3.pytree import convert
from modules.helper import *
from modules.preprocessing import * 
import os
import time
from prettytable import PrettyTable
###########################################################
#                                                         #
#         CRYSTAL PLASTICITY PARAMETER CALIBRATION        #
#   Tools required: DAMASK and Finnish Supercomputer CSC  #
#                                                         #
###########################################################

# -------------------------------------------------------------------
#   Stage 0: Choose the CP model, the optimization algorithm, number of initial simulations,
#   the target curve index to fit, project path folder and material name
# -------------------------------------------------------------------

dataConfig = pd.read_excel("optimize_config.xlsx", nrows= 1, engine="openpyxl")
dataConfig = dataConfig.T.to_dict()[0]
# print(dataConfig)

material = dataConfig["material"]

server = dataConfig["server"]

CPLaw = dataConfig["CPLaw"]

optimizerName = dataConfig["optimizerName"] 

# The 8 types of loadings: 2 linear loadings and 6 nonlinear loadings
loadings = [
            #"linear_uniaxial_RD", 
            "linear_uniaxial_TD",
            #"nonlinear_biaxial_RD", 
            #"nonlinear_biaxial_TD",     
            #"nonlinear_planestrain_RD",     
            #"nonlinear_planestrain_TD",     
            #"nonlinear_uniaxial_RD", 
            #"nonlinear_uniaxial_TD"
            ]

# loadings = [
#             "linear_uniaxial_RD", 
#             #"linear_uniaxial_TD",
#             "nonlinear_biaxial_RD", 
#             "nonlinear_biaxial_TD",     
#             "nonlinear_planestrain_RD",     
#             "nonlinear_planestrain_TD",     
#             "nonlinear_uniaxial_RD", 
#             "nonlinear_uniaxial_TD"
#             ]

exampleLoading = loadings[0]

yieldingPoints = {
    "PH": 
    {
        "linear_uniaxial_RD": 0.004,
        "nonlinear_biaxial_RD": 0.118,
        "nonlinear_biaxial_TD": 0.118,
        "nonlinear_planestrain_RD": 0.092,
        "nonlinear_planestrain_TD": 0.092,
        "nonlinear_uniaxial_RD": 0.061, 
        "nonlinear_uniaxial_TD": 0.061,
    },
    
    "DB": 
    {
        "linear_uniaxial_RD": 0.004,
        "nonlinear_biaxial_RD": 0.118,
        "nonlinear_biaxial_TD": 0.118,
        "nonlinear_planestrain_RD": 0.091,
        "nonlinear_planestrain_TD": 0.091,
        "nonlinear_uniaxial_RD": 0.060, 
        "nonlinear_uniaxial_TD": 0.060,
    },
}

convertUnit = 1e-6

def printList(messages):
    for message in messages:
        print(message)

initialSims = dataConfig["initialSims"]  

method = dataConfig["method"]  

curveIndex = dataConfig["curveIndex"] 

searchingSpace = dataConfig["searchingSpace"]  
 
roundContinuousDecimals = dataConfig["roundContinuousDecimals"]  

linearYieldingDevGlobal = dataConfig["linearYieldingDev"]

largeLinearHardeningDevGlobal = dataConfig["largeLinearHardeningDev"]  

smallLinearHardeningDevGlobal = dataConfig["smallLinearHardeningDev"]  

largeNonlinearHardeningDevGlobal = dataConfig["largeNonlinearHardeningDev"]  

smallNonlinearHardeningDevGlobal = dataConfig["smallNonlinearHardeningDev"]  

# Setting the weights of the two yield stress objective functions:  
weightsYielding = {"wy1": 0.0995, "wy2": 0.0005}

# Setting the weights of the two hardening objective functions:  
weightsHardening = {"wh1": 0.099, "wh2": 0.001}

# Setting the weights of the seven loadings in the yielding and hardening fitness function:  

weightsLoading = {
    "linear_uniaxial_RD": 0.19, 
    "nonlinear_biaxial_RD": 0.06, 
    "nonlinear_biaxial_TD": 0.17,     
    "nonlinear_planestrain_RD": 0.04,     
    "nonlinear_planestrain_TD": 0.15,     
    "nonlinear_uniaxial_RD": 0.01, 
    "nonlinear_uniaxial_TD": 0.38,
}

paramsFormatted = {
    "PH": {
        "a": "a", 
        "gdot0": "γ̇₀", 
        "h0": "h₀", 
        "n": "n", 
        "tau0": "τ₀", 
        "tausat": "τₛₐₜ",
        "self": "self", 
        "coplanar": "coplanar", 
        "collinear": "collinear", 
        "orthogonal": "orthogonal", 
        "glissile": "glissile", 
        "sessile": "sessile", 
    },
    "DB": {
        "dipmin": "d̂α", 
        "islip": "iₛₗᵢₚ", 
        "omega": "Ω", 
        "p": "p", 
        "q": "q", 
        "tausol": "τₛₒₗ",
        "Qs": "Qs",
        "Qc": "Qc",
        "v0": "v₀",
        "rho_e": "ρe",
        "rho_d": "ρd",   
        "D0": "D0",
        "self": "self", 
        "coplanar": "coplanar", 
        "collinear": "collinear", 
        "orthogonal": "orthogonal", 
        "glissile": "glissile", 
        "sessile": "sessile", 
    },
}

paramsUnit = {
    "PH": {
        "a": "", # Empty string means this parameter is dimensionless
        "gdot0": "s⁻¹", 
        "h0": "Pa", 
        "n": "", 
        "tau0": "Pa", 
        "tausat": "Pa",
        "self": "", 
        "coplanar": "", 
        "collinear": "", 
        "orthogonal": "", 
        "glissile": "", 
        "sessile": "", 
    },
    "DB": {
        "dipmin": "b", 
        "islip": "", 
        "omega": "b³", 
        "p": "", 
        "q": "", 
        "tausol": "Pa",
        "Qs": "J",
        "Qc": "J",
        "v0": "m/s",
        "rho_e": "m⁻²",
        "rho_d": "m⁻²",   
    },
}


# The project path folder
projectPath = os.getcwd()

###############################################
#  Printing the configurations to the console #
###############################################

if CPLaw == "PH":
    law = "phenomenological law"
elif CPLaw == "DB":
    law = "dislocation-based law"

target_curve = f"{CPLaw}{curveIndex}"

linearYieldingDevGlobalPercent = f"{linearYieldingDevGlobal}%"

largeLinearHardeningDevGlobalPercent = f"{largeLinearHardeningDevGlobal}%"

smallLinearHardeningDevGlobalPercent = f"{smallLinearHardeningDevGlobal}%"

largeNonlinearHardeningDevGlobalPercent = f"{largeNonlinearHardeningDevGlobal}%"

smallNonlinearHardeningDevGlobalPercent = f"{smallNonlinearHardeningDevGlobal}%"

configMessages = [  
    f"\nWelcome to the Crystal Plasticity Parameter Calibration software\n\n",
    f"The configurations you have chosen: \n",       
]

logTable = PrettyTable()

logTable.field_names = ["Configurations", "User choice"]
logTable.add_row(["Server", server])
logTable.add_row(["Initial simulation number", initialSims])
logTable.add_row(["Initial simulation method", method])
logTable.add_row(["Searching space", searchingSpace])
logTable.add_row(["Material", material])
logTable.add_row(["CP Law", CPLaw])
logTable.add_row(["Optimizer", optimizerName])
logTable.add_row(["Target curves", target_curve])
logTable.add_row(["Rounding decimals", roundContinuousDecimals])
logTable.add_row(["Linear yielding dev", linearYieldingDevGlobalPercent])
logTable.add_row(["Large linear hardening dev", largeLinearHardeningDevGlobalPercent])
logTable.add_row(["Small linear hardening dev", smallLinearHardeningDevGlobalPercent])
logTable.add_row(["Large nonlinear hardening dev", largeNonlinearHardeningDevGlobalPercent])
logTable.add_row(["Small nonlinear hardening dev", smallNonlinearHardeningDevGlobalPercent])

configMessages.append(logTable.get_string())
configMessages.append("\n")

#########################################################
# Creating necessary directories for the configurations #
#########################################################

def checkCreate(path):
    if not os.path.exists(path):
        os.mkdir(path)

# For log
checkCreate("log")
path = f"log/{material}"
checkCreate(path)
checkCreate(f"{path}/{CPLaw}")

# For manualParams
checkCreate("manualParams")
path = f"manualParams/{material}"
checkCreate(path)
checkCreate(f"{path}/{CPLaw}")

# For parameter_analysis
checkCreate("parameter_analysis")
path = f"parameter_analysis/{material}"
checkCreate(path)
checkCreate(f"{path}/{CPLaw}")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/{loading}")

# For results 
checkCreate("results")
path = f"results/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")

checkCreate(f"{path}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}")
checkCreate(f"{path}/{CPLaw}/universal")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/universal/{loading}")

# For simulations

checkCreate("simulations")
path = f"simulations/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")

checkCreate(f"{path}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}/{loading}")
checkCreate(f"{path}/{CPLaw}/universal")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/universal/{loading}")  

# For targets 
checkCreate("targets")
path = f"targets/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/{loading}")
checkCreate(f"{path}/{CPLaw}/param_info")

# For templates
checkCreate("templates")
path = f"templates/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/{loading}")
checkCreate(f"{path}/{CPLaw}/param_info")


# Loading the parameter information
#                                                                           
getParamRanges(material, CPLaw, curveIndex, searchingSpace, roundContinuousDecimals)
general_param_info = loadGeneralParam(material, CPLaw)

# print(general_param_info)
# param_infos_GA_discrete = param_infos_GA_discrete_func(param_infos) # For GA discrete
# param_infos_GA_continuous = param_infos_GA_continuous_func(param_infos) # For GA continuous
# param_infos_BO = param_infos_BO_func(param_infos) # For both BO discrete and continuous
# param_infos_PSO_low = param_infos_PSO_low_func(param_infos)
# param_infos_PSO_high = param_infos_PSO_high_func(param_infos)

#print(general_param_info)
# print(param_infos)
info = {
    'param_info': general_param_info,
    'server': server,
    'loadings': loadings,
    'CPLaw': CPLaw,
    'initialSims': initialSims,
    'projectPath': projectPath,
    'optimizerName': optimizerName,
    'material': material,
    'method': method,
    'searchingSpace': searchingSpace,
    'roundContinuousDecimals': roundContinuousDecimals,
    'loadings': loadings,
    'exampleLoading': exampleLoading
}

printList(configMessages)


assert largeLinearHardeningDevGlobal >= linearYieldingDevGlobal, "largeLinearHardeningDev must be larger than or equal linearYieldingDev"

assert largeLinearHardeningDevGlobal > smallLinearHardeningDevGlobal, "largeLinearHardeningDev must be larger than smallLinearHardeningDev"

assert largeNonlinearHardeningDevGlobal > smallNonlinearHardeningDevGlobal, "largeNonlinearHardeningDev must be larger than smallNonlinearHardeningDev"

assert smallNonlinearHardeningDevGlobal > smallLinearHardeningDevGlobal, "smallNonlinearHardeningDev must be larger than smallLinearHardeningDev"

print("Generating necessary directories\n")
print(f"The path to your main project folder is\n", f"{projectPath}\n\n")

# ANN model selection from smallest test error 
numberOfHiddenLayers = 2
hiddenNodesFormula = "formula1"
ANNOptimizer = "Adam"
L2_regularization = 0.5
learning_rate = 0.05

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
        "linear_uniaxial_RD": 2400, 
        "nonlinear_biaxial_RD": 2400, 
        "nonlinear_biaxial_TD": 2400,     
        "nonlinear_planestrain_RD": 2400,     
        "nonlinear_planestrain_TD": 2400,     
        "nonlinear_uniaxial_RD": 2400, 
        "nonlinear_uniaxial_TD": 2400
    }
}
