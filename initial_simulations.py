#############################################
#                                           #
#        Running initial simulations        #
#                                           #
#############################################

def main():

    ############################################
    # Generating universal initial simulations #
    ############################################

    print("\n" + 70 * "*" + "\n\n")
    print(f"Step 0: Running initial simulations\n\n")

    simUniversal = SIM(info)
    if method == "manual":
        print("Starting initial simulations\n")
        manualParams = np.load(f"manualParams/{material}/{CPLaw}/initial_params.npy", allow_pickle=True)
        tupleParams = manualParams[0:25] # <-- Run the parameters in small batches
        simUniversal.run_initial_simulations(tupleParams)
        print(f"Done. {len(tupleParams)} simulations completed.")
    elif method == "auto":
        if not os.path.exists(f"results/{material}/{CPLaw}/universal/initial_processCurves.npy"):
            print("Starting initial simulations\n")
            simUniversal.run_initial_simulations()
        initial_processCurves = np.load(f'results/{material}/{CPLaw}/universal/initial_processCurves.npy', allow_pickle=True).tolist()
        print(f"Initial simulations finished. {len(initial_processCurves['linear_uniaxial_RD'])} simulations completed\n")

if __name__ == '__main__':
    # External libraries
    import os
    import numpy as np
    from optimize_config import *
    from modules.SIM_damask2 import *
    from modules.preprocessing import *
    from modules.helper import *
    main()
