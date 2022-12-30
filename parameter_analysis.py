
def main():
    baseParams = {
        "PH": {
            "a": 1.5, 
            "gdot0": 2, 
            "h0": 500, 
            "n": 30, 
            "tau0": 50, 
            "tausat": 150,
            "self": 1, 
            "coplanar": 1, 
            "collinear": 1, 
            "orthogonal": 1, 
            "glissile": 1, 
            "sessile": 1,
        },
        "DB": {
            "dipole": 7, 
            "islip": 50, 
            "omega": 25, 
            "p": 0.5, 
            "q": 1.5, 
            "tausol": 70,
            "Qs": 3,
            "Qc": 2,
            "v0": 15,
            "rho_e": 3,
            "rho_d": 500,       
        }
    }
    
    # Should be an odd number
    numberOfAnalysisCurves = 5

    stepParams = {
        "PH": {
            "a": 0.2, 
            "gdot0": 0.5, 
            "h0": 100, 
            "n": 5, 
            "tau0": 5, 
            "tausat": 10,
            "self": 0.4, 
            "coplanar": 0.4, 
            "collinear": 0.4, 
            "orthogonal": 0.4, 
            "glissile": 0.4, 
            "sessile": 0.4, 
        },
        "DB": {
            "dipole": 2, 
            "islip": 5, 
            "omega": 10, 
            "p": 0.1, 
            "q": 0.2, 
            "tausol": 20,
            "Qs": 1,
            "Qc": 0.1,
            "v0": 5,
            "rho_e": 1,
            "rho_d": 200,   
        },

    }
    
    getParamRanges(material, CPLaw, curveIndices, searchingSpace, searchingType, roundContinuousDecimals, 3, 9, 1)
    general_param_info = loadGeneralParam(material, CPLaw)
    for param in general_param_info:
        if param in baseParams[CPLaw].keys():
            general_param_info[param]["optimized_target"] = True
        else: 
            general_param_info[param]["optimized_target"] = False
    print(general_param_info)

    rounding = 2
    info = {
        'hyperqueue': hyperqueue,
        'loadings': loadings,
        'server': server,
        'param_info': general_param_info,
        'CPLaw': CPLaw,
        'initialSims': initialSims,
        'projectPath': projectPath,
        'optimizerName': optimizerName,
        'material': material,
        'method': method,
        'searchingSpace': searchingSpace,
        'rounding': rounding,
        'roundContinuousDecimals': roundContinuousDecimals,
    }

    paramNames = {
        "PH": list(baseParams["PH"].keys()),
        "DB": list(baseParams["DB"].keys()),
    }
    def checkCreate(path):
        if not os.path.exists(path):
            os.mkdir(path)

    # For parameter_analysis
    path = f"parameter_analysis/{material}"
    checkCreate(path)
    print(CPLaw)
    checkCreate(f"{path}/{CPLaw}")
    checkCreate(f"{path}/{CPLaw}/results")
    for loading in loadings:
        checkCreate(f"{path}/{CPLaw}/{loading}")
        for param in paramNames[CPLaw]:
            checkCreate(f"{path}/{CPLaw}/{loading}/{param}")
        checkCreate(f"{path}/{CPLaw}/{loading}/base")
    sim = SIM(info)
    sim.run_parameter_analysis(baseParams[CPLaw], stepParams[CPLaw], numberOfAnalysisCurves, paramNames[CPLaw])
    print(f"Done. Parameter analysis completed.")


if __name__ == '__main__':
    # External libraries
    from modules.SIM import *
    from modules.preprocessing import *
    from modules.stoploss import *
    from modules.helper import *
    import os
    from optimize_config import *
    main()
# python parameter_analysis.py