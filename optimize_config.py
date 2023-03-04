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

def main_config():
    dataConfig = pd.read_excel("optimize_config.xlsx", nrows= 1, engine="openpyxl")
    dataConfig = dataConfig.T.to_dict()[0]

    material = dataConfig["material"]

    server = dataConfig["server"]

    CPLaw = dataConfig["CPLaw"]

    optimizerName = dataConfig["optimizerName"] 

    # The 8 types of loadings: 2 linear loadings and 6 nonlinear loadings
    # loadings = [
    #             #"linear_uniaxial_RD", 
    #             "linear_uniaxial_TD",
    #             #"nonlinear_biaxial_RD", 
    #             #"nonlinear_biaxial_TD",     
    #             #"nonlinear_planestrain_RD",     
    #             #"nonlinear_planestrain_TD",     
    #             #"nonlinear_uniaxial_RD", 
    #             #"nonlinear_uniaxial_TD"
    #             ]

    loadings = [
                "linear_uniaxial_RD", 
                "linear_uniaxial_TD",
                "nonlinear_biaxial_RD", 
                "nonlinear_biaxial_TD",     
                "nonlinear_planestrain_RD",     
                "nonlinear_planestrain_TD",     
                "nonlinear_uniaxial_RD", 
                "nonlinear_uniaxial_TD"
                ]

    exampleLoading = loadings[0]

    yieldingPoints = {
        "PH": 
        {
            "linear_uniaxial_RD": 0.004,
            "linear_uniaxial_TD": 0.004,
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
            "linear_uniaxial_TD": 0.004,
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

    linearYieldingDev = dataConfig["linearYieldingDev"]

    firstLinearHardeningDev = dataConfig["firstLinearHardeningDev"]  

    secondLinearHardeningDev = dataConfig["secondLinearHardeningDev"]  

    firstNonlinearHardeningDev = dataConfig["firstNonlinearHardeningDev"]  

    secondNonlinearHardeningDev = dataConfig["secondNonlinearHardeningDev"]  

    # Setting the weights of the two yield stress objective functions:  
    weightsYielding = {"wy1": 0.0995, "wy2": 0.0005}

    # Setting the weights of the two hardening objective functions:  
    weightsHardening = {"wh1": 0.099, "wh2": 0.001}

    # Setting the weights of the seven loadings in the yielding and hardening fitness function:  

    weightsLoading = {
        "linear_uniaxial_RD": 0.10, 
        "linear_uniaxial_TD": 0.09,
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

    linearYieldingDevPercent = f"{linearYieldingDev}%"

    firstLinearHardeningDevPercent = f"{firstLinearHardeningDev}%"

    secondLinearHardeningDevPercent = f"{secondLinearHardeningDev}%"

    firstNonlinearHardeningDevPercent = f"{firstNonlinearHardeningDev}%"

    secondNonlinearHardeningDevPercent = f"{secondNonlinearHardeningDev}%"

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
    logTable.add_row(["CP Law", law])
    logTable.add_row(["Optimizer", optimizerName])
    logTable.add_row(["Target curves", target_curve])
    logTable.add_row(["Rounding decimals", roundContinuousDecimals])
    logTable.add_row(["Linear yielding dev", linearYieldingDevPercent])
    logTable.add_row(["Large linear hardening dev", firstLinearHardeningDevPercent])
    logTable.add_row(["Small linear hardening dev", secondLinearHardeningDevPercent])
    logTable.add_row(["Large nonlinear hardening dev", firstNonlinearHardeningDevPercent])
    logTable.add_row(["Small nonlinear hardening dev", secondNonlinearHardeningDevPercent])

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
    checkCreate(f"{path}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}/common")
    for loading in loadings:
        checkCreate(f"{path}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}/{loading}")
    
    checkCreate(f"{path}/{CPLaw}/universal")
    checkCreate(f"{path}/{CPLaw}/universal/common")
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
    param_info = loadGeneralParam(material, CPLaw)


    printList(configMessages)


    assert firstLinearHardeningDev >= linearYieldingDev, "largeLinearHardeningDev must be larger than or equal linearYieldingDev"

    assert firstLinearHardeningDev > secondLinearHardeningDev, "largeLinearHardeningDev must be larger than smallLinearHardeningDev"

    assert firstNonlinearHardeningDev > secondNonlinearHardeningDev, "largeNonlinearHardeningDev must be larger than smallNonlinearHardeningDev"

    assert secondNonlinearHardeningDev > secondLinearHardeningDev, "smallNonlinearHardeningDev must be larger than smallLinearHardeningDev"

    print("Generating necessary directories\n")
    print(f"The path to your main project folder is\n", f"{projectPath}\n")

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

    info = {
        'param_info': param_info,
        'server': server,
        'loadings': loadings,
        'CPLaw': CPLaw,
        'convertUnit': convertUnit,
        'initialSims': initialSims,
        'curveIndex': curveIndex,
        'projectPath': projectPath,
        'optimizerName': optimizerName,
        'param_info': param_info,
        'material': material,
        'method': method,
        'searchingSpace': searchingSpace,
        'roundContinuousDecimals': roundContinuousDecimals,
        'linearYieldingDev': linearYieldingDev, 
        'firstLinearHardeningDev': firstLinearHardeningDev, 
        'secondLinearHardeningDev': secondLinearHardeningDev, 
        'firstNonlinearHardeningDev': firstNonlinearHardeningDev,
        'secondNonlinearHardeningDev': secondNonlinearHardeningDev,
        'loadings': loadings,
        'exampleLoading': exampleLoading,
        'yieldingPoints': yieldingPoints, 
        'weightsYielding': weightsYielding,
        'weightsHardening': weightsHardening,
        'weightsLoading':  weightsLoading,
        'paramsFormatted': paramsFormatted,
        'paramsUnit': paramsUnit,
        'numberOfHiddenLayers': numberOfHiddenLayers,
        'hiddenNodesFormula': hiddenNodesFormula,
        'ANNOptimizer': ANNOptimizer,
        'L2_regularization': L2_regularization,
        'learning_rate': learning_rate,
        'loading_epochs': loading_epochs,
    }

    return info
