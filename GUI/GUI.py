def main():
    st.title('Crystal plasticity Application')
    st.text("Author: Nguyen Xuan Binh \nInstitution: Aalto University \nCourse: Computational Engineering Project")
    st.markdown("This is an online tool that plots stress-strain curves in the crystal plasticity model and analyzes the fitting parameter optimization. Crystal plasticity studies the plastic deformation of polycrystalline materials")
    st.image("GUI/pictures/CP_illustration.png")

    # Using "with" notation
    with st.sidebar:
        st.header('Please specify your choice')
        #########
        materials = ("RVE_1_40_D", "512grains512")
        material = st.radio("Please select the material", materials)
        #########
        CPLaws = ("PH", "DB")
        CPLaw = st.radio("Please select the crystal plasticity law", CPLaws)
        if CPLaw == "PH":
            stressUnit = "MPa"
            convertUnit = 1
        elif CPLaw == "DB":
            stressUnit = "Pa"
            convertUnit = 1e-6
        #########
        curveIndex = st.text_input('Please select the curve index', '4')
        #########
        optimizerNames = ("GA", "BO", "PSO")
        optimizerName = st.radio("Please select the optimizer", optimizerNames)
        #########
        roundContinuousDecimals = st.text_input('Number of rounding decimals', 4)
        roundContinuousDecimals = int(roundContinuousDecimals)
        #########
        searchingSpaces = ("discrete", "continuous")
        searchingSpace = st.radio("Please select the searching space", searchingSpaces)
        if stressUnit == "MPa":
            convertUnit = 1
        else:
            convertUnit = 1e-6

    # All common data 

    loadings = ["linear_uniaxial_RD", 
                "nonlinear_biaxial_RD", 
                "nonlinear_biaxial_TD",
                "nonlinear_planestrain_RD",  
                "nonlinear_planestrain_TD",
                "nonlinear_uniaxial_RD",   
                "nonlinear_uniaxial_TD"]

    yieldingPoints = {
        "PH": 
        {
            "linear_uniaxial_RD": 0.004,
            "nonlinear_biaxial_RD": 0.118,
            "nonlinear_biaxial_TD": 0.118,
            "nonlinear_planestrain_RD": 0.091,
            "nonlinear_planestrain_TD": 0.091,
            "nonlinear_uniaxial_RD": 0.060, 
            "nonlinear_uniaxial_TD": 0.060,
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
            "dipole": "dα", 
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
        },
    }

    parameterRows = {
        "PH":
            [r"$a$", 
            r"$h_0$", 
            r"$\tau_0$", 
            r"$\tau_{sat}$", 
            r"$self$", 
            r"$coplanar$", 
            r"$collinear$", 
            r"$orthogonal$", 
            r"$glissile$", 
            r"$sessile$"],
        "DB": 
            [r"$d^\alpha$", 
            r"$i_{slip}$", 
            r"$\Omega$", 
            r"$p$", 
            r"$q$", 
            r"$\tau_{sol}$", 
            r"$Q_s$", 
            r"$Q_c$", 
            r"$v_0$", 
            r"$\rho_e$"]
    }
       



    standardColors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    
    # Starting the tabs
    tab1, tab2, tab3, tab4= st.tabs(["A. Preprocessing", "B. Simulation results", "C. Plotting MSE", "D. Parameter analysis"])

    with tab1:

        st.header('Preprocessing stage')
    
        st.markdown("Please select the curve types that you want to plot")

        targetTrueCheck = st.checkbox("Plot true curve", value=True)
        targetProcessedCheck = st.checkbox("Plot processed curve", value=True)
        yieldCheck = st.checkbox("Plot yielding point", value=True)
        initialSimCheck = st.checkbox("Plot initial simulations", value=False)
        expTypes = ("Experimental curve", "DAMASK simulated curve")
        expType = st.radio("Please select the experimental curve type", expTypes)
        initialSimTypes = ("True curves", "Processed curves")
        initialSimType = st.radio("Please select the initial simulation curve type", initialSimTypes)
        for loading in loadings:
            title = ""
            size = 15
            figure(figsize=(6, 4))
            if initialSimCheck:
                if initialSimType == "True curves":
                    initial_data = np.load(f'results/{material}/{CPLaw}/universal/initial_trueCurves.npy', allow_pickle=True)
                elif initialSimType == "Processed curves":
                    initial_data = np.load(f'results/{material}/{CPLaw}/universal/initial_processCurves.npy', allow_pickle=True)
                initial_data = initial_data.tolist()
                for curve in initial_data[loading].values():
                    strain = curve["strain"] 
                    stress = curve["stress"] * convertUnit
                    plt.plot(strain, stress, c='orange', alpha=0.07)
                plt.plot(strain, stress, label = f"Initial simulations x 500",c='orange', alpha=0.3)
                title += f" | Universal initial simulations\n({CPLaw} law)"
            if expType == "Experimental curve":
                currentPath = f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}.xlsx"
                if targetTrueCheck:
                    trueCurve = preprocessExperimentalTrue(currentPath, True)
                    trueStrain = trueCurve["strain"]
                    trueStress = trueCurve["stress"] * convertUnit
                    plt.plot(trueStrain, trueStress, c='blue', label="Experimental", alpha = 1)
                if targetProcessedCheck:
                    processCurve = preprocessExperimentalFitted(currentPath, True)
                    processStrain = processCurve["strain"]
                    processStress = processCurve["stress"] * convertUnit
                    plt.plot(processStrain, processStress, c='black', label="Swift - Voce fitting")
                title += f"Target curve " 
            if expType == "DAMASK simulated curve":
                currentPath = f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}.txt"
                if targetTrueCheck:
                    trueCurve = preprocessDAMASKTrue(currentPath)
                    trueStrain = trueCurve["strain"]
                    trueStress = trueCurve["stress"] * convertUnit
                    plt.plot(trueStrain, trueStress, c='blue', label="True curve")
                if targetProcessedCheck:
                    if loading == "linear_uniaxial_RD":
                        processCurve = preprocessDAMASKLinear(currentPath)
                    else:
                        processCurve = preprocessDAMASKNonlinear(currentPath)
                    processStrain = processCurve["strain"]
                    processStress = processCurve["stress"] * convertUnit
                    plt.plot(processStrain, processStress, c='black', label="Processed curve")
                title += f"experimental curve "
            if yieldCheck:
                yieldingPoint = yieldingPoints[CPLaw][loading]
                plt.axvline(x = yieldingPoint, color = 'black', label = f"Yielding point = {yieldingPoint}", alpha=0.5)
            
            title += f"\n({loading})"
            #plt.title(f"{loading} | {CPLaw} model" , size=size + 4)
            plt.title(f"{loading}" , size=size + 3)
            
            plt.xticks(fontsize=size-2)
            plt.yticks(fontsize=size-2)
            if CPLaw == "PH": 
                plt.xlim([0, 0.27])
                plt.ylim([0, 1750])
                plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
                plt.yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750])
            elif CPLaw == "DB":
                plt.xlim([0, 0.27])
                plt.ylim([0, 1000])
                plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
                plt.yticks([0, 250, 500, 750, 1000])
            plt.xlim([0, 0.27])
            plt.ylim([0, 400])
            plt.xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
            plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450])
            plt.ylabel('True stress, MPa', size=size + 1)
            plt.xlabel("True strain, -", size=size + 1)
            legend = plt.legend(loc=2, frameon=False, fontsize=size-2, ncol=1, facecolor='white')
            legend.get_frame().set_linewidth(0.0)
            st.pyplot(plt)


    with tab2:
        st.header('Simulation result stage')

        stageNumbers = ("All stages", "default parameters", "1st stage", "2nd stage", "3rd stage", "4th stage")
        stageNumber = st.radio("Please select the simulation stage", stageNumbers)

        st.subheader("Plot all stages")

        curveTypes = ("True curves", "Processed curves", "Interpolated curves")
        curveType = st.radio("Please select the curve type", curveTypes)
        if curveType == "True curves":
            curveType = "true"
        if curveType == "Processed curves":
            curveType = "process"
        if curveType == "Interpolated curves":
            curveType = "interpolate"
        if curveType == "interpolate" and CPLaw == "DB":
            convertUnit = 1
        plotDirection = ("vertical", "horizontal")
        plotDirection = st.radio("Please select the plotting direction", plotDirection)

        if plotDirection == "vertical":
            indexLoading = {
                "linear_uniaxial_RD": (0,0), 
                "tableParam": (0,1),
                "nonlinear_biaxial_RD": (1,0), 
                "nonlinear_biaxial_TD": (1,1),     
                "nonlinear_planestrain_RD": (2,0),     
                "nonlinear_planestrain_TD": (2,1),     
                "nonlinear_uniaxial_RD": (3,0), 
                "nonlinear_uniaxial_TD": (3,1),
            }
        else:
            indexLoading = {
                "linear_uniaxial_RD": (0,0), 
                "tableParam": (1,0),
                "nonlinear_biaxial_RD": (0,1), 
                "nonlinear_biaxial_TD": (1,1),     
                "nonlinear_planestrain_RD": (0,2),     
                "nonlinear_planestrain_TD": (1,2),     
                "nonlinear_uniaxial_RD": (0,3), 
                "nonlinear_uniaxial_TD": (1,3),
            }

        resultPath = f"results/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}"


        if plotDirection == "vertical":
            fig, ax = plt.subplots(4,2, figsize=(20,28), constrained_layout=True)
        if plotDirection == "horizontal":
            fig, ax = plt.subplots(2,4, figsize=(30,12), constrained_layout=True)
        # fig.tight_layout(rect=[0, 0.03, 1, 0.9])
        # fig.subplots_adjust(top=0.95)
        
        fig.suptitle(f'Results for the experimental curve | {CPLaw} model\nOptimizer: {optimizerName} | Searching space : {searchingSpace}', fontsize=35)
        
        if stageNumber == "All stages":

            # Extracting the results

            stage_CurvesList = np.load(f"{resultPath}/stage_CurvesList.npy", allow_pickle=True).tolist()
            
            col2a, col2b = st.columns(2)
            with col2a:
                startingIter = st.number_input("Starting iteration", min_value=-len(stage_CurvesList), max_value=len(stage_CurvesList), value=-2, key="Result1")
                #startingIter = st.number_input("Starting iteration", value=23)
            with col2b:
                endingIter = st.number_input("Ending iteration", min_value=-len(stage_CurvesList), max_value=len(stage_CurvesList), value=-1, key="Result2")
                #endingIter = st.number_input("Ending iteration", value=26)
   

            stage_CurvesList = stage_CurvesList[startingIter-1:endingIter]
            #stage_CurvesList = [stage_CurvesList[67]]
            parameterValues = list([stageCurves["parameters_tuple"] for stageCurves in stage_CurvesList])
            #st.write(stage_CurvesList)
            iterationColumns = []
            paramValues2D = []

            numberOfIterations = len(parameterValues) 
            repeatedCycles = math.ceil(numberOfIterations/10) 
            columnColors = standardColors * repeatedCycles


            for iteration in range(1, len(stage_CurvesList) + 1):
                iterationColumns.append(f"Iter {iteration}")

            for tupleParams in parameterValues:
                paramValues = []
                dictParams = dict(tupleParams)
                for param in dictParams:
                    paramValues.append(round(dictParams[param], roundContinuousDecimals))
                paramValues2D.append(paramValues)

            # transposing the matrix
            paramValues2D = np.array(paramValues2D).T
            
            size = 28

            for loading in loadings:
                iteration = 1
                pathTarget = f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_{curveType}.npy"
                target_Curve = np.load(pathTarget, allow_pickle=True).tolist()
                i = indexLoading[loading][0]
                j = indexLoading[loading][1]
                ax[i][j].plot(target_Curve["strain"], target_Curve["stress"] * convertUnit, color = "k", linewidth=3, alpha=1, label=f"Target\ncurve")

                for stageCurves in stage_CurvesList:
                    ax[i][j].plot(stageCurves[curveType][loading]["strain"], stageCurves[curveType][loading]["stress"] * convertUnit,  linewidth=3, alpha=1, label=f"Iter {iteration}")
                    iteration += 1
                    
                    ax[i][j].set_xlim(right = 0.27)
                    ax[i][j].set_ylim(top = 370)
                    ax[i][j].tick_params(axis='x', labelsize= size)
                    ax[i][j].tick_params(axis='y', labelsize= size)
                    ax[i][j].set_ylabel('Stress, MPa', size= size)
                    ax[i][j].set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
                    ax[i][j].set_xlabel("Strain, -", size= size)
                    ax[i][j].set_title(f"{loading}", size= 5/4 * size)
                    legend = ax[i][j].legend(loc=4, frameon=False, fontsize= size-2, ncol=2) #, shadow =True, framealpha=1)
                    # plt.grid()
                    legend.get_frame().set_linewidth(0.0)

            iTable = indexLoading["tableParam"][0]
            jTable = indexLoading["tableParam"][1]
            ax[iTable][jTable].axis('tight')
            ax[iTable][jTable].axis('off')
            table = ax[iTable][jTable].table(cellText=paramValues2D, 
                                    colLabels=iterationColumns, 
                                    rowLabels=parameterRows[CPLaw], 
                                    loc='upper center', 
                                    cellLoc='center', 
                                    colLoc="center",
                                    rowLoc="center",
                                    colWidths=[0.2 for x in iterationColumns],
                                    colColours= columnColors, 
                                    fontsize=40)
            #ax[iTable][jTable].set_title(f"Parameter values", size= 5/4 * size)
            table.auto_set_column_width(col=iterationColumns)
            table.auto_set_font_size(False)
            table.set_fontsize(25)
            table.scale(3, 3)

            st.pyplot(plt)

            # size = 28

            # for loading in loadings:
            #     iteration = 1
            #     pathTarget = f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_{curveType}.npy"
            #     target_Curve = np.load(pathTarget, allow_pickle=True).tolist()
            #     i = indexLoading[loading][0]
            #     j = indexLoading[loading][1]
            #     ax[i][j].plot(target_Curve["strain"], target_Curve["stress"] * convertUnit, color = "k", linewidth=3, alpha=1, label=f"Exp.\ncurve")
      
            #     ax[i][j].plot(stage_curves[curveType][loading]["strain"], stage_curves[curveType][loading]["stress"] * convertUnit,  linewidth=3, alpha=1, label=f"Sim.\ncurve")
            #     iteration += 1
                
            #     ax[i][j].set_xlim(right = 0.27)
            #     ax[i][j].set_ylim(top = 370)
            #     ax[i][j].tick_params(axis='x', labelsize= size)
            #     ax[i][j].tick_params(axis='y', labelsize= size)
            #     ax[i][j].set_ylabel('Stress, MPa', size= size)
            #     ax[i][j].set_xlabel('Strain, -', size= size)
            #     ax[i][j].set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
            #     ax[i][j].set_title(f"{loading} loading", size= 5/4 * size)
            #     legend = ax[i][j].legend(loc=4, frameon=False, fontsize= size - 2, ncol=2) #, shadow =True, framealpha=1)
            #     # plt.grid()
            #     legend.get_frame().set_linewidth(0.0)
            # iTable = indexLoading["tableParam"][0]
            # jTable = indexLoading["tableParam"][1]
            # ax[iTable][jTable].axis('tight')
            # ax[iTable][jTable].axis('off')
            # table = ax[iTable][jTable].table(cellText=paramValues2D, 
            #                         colLabels=iterationColumns, 
            #                         rowLabels=parameterRows[CPLaw], 
            #                         loc='upper center', 
            #                         cellLoc='center', 
            #                         colLoc="center",
            #                         rowLoc="center",
            #                         colWidths=[0.2 for x in iterationColumns], # Change this for column width
            #                         colColours= columnColors, 
            #                         rowColours= rowColors,
            #                         fontsize=40)
            # #ax[iTable][jTable].set_title(f"Parameter values", size= 5/4 * size)
            # table.auto_set_column_width(col=iterationColumns)
            # table.auto_set_font_size(True)
            # table.set_fontsize(25)
            # table.scale(3, 3) # Scaling the table
        else:
            parameterTypes = {
                "PH": {
                    "default parameters": [],
                    "large_yieldingParams":  ["tau0"],
                    "small_yieldingParams": [],
                    "large_hardeningParams": ["a", "h0", "tausat"],
                    "small_hardeningParams": ["self", "coplanar", "collinear", "orthogonal", "glissile", "sessile"]
                },
                "DB": {
                    "default parameters": [],
                    "large_yieldingParams":  ["tausol", "Qs", "rho_e"],
                    "small_yieldingParams": ["p", "q", "v0"],
                    "large_hardeningParams": ["dipole", "islip", "omega", "Qc"],
                    "small_hardeningParams": []
                }
            }
            if stageNumber == "default parameters":
                stage_curves = np.load(f"{resultPath}/default_curves.npy", allow_pickle=True).tolist()
                targetParams = parameterTypes[CPLaw]["default parameters"]
                fig.suptitle(f'Default parameters for experimental curves | {CPLaw} model\nOptimizer: {optimizerName} | Searching space : {searchingSpace}', fontsize=35)
                color = "white"
            if stageNumber == "1st stage":
                stage_curves = np.load(f"{resultPath}/stage1_curves.npy", allow_pickle=True).tolist()
                targetParams = parameterTypes[CPLaw]["large_yieldingParams"]
                fig.suptitle(f'1st stage large yielding parameters calibration result | {CPLaw} model\nOptimizer: {optimizerName} | Searching space : {searchingSpace} | Allowed Δᵧ_ₗᵢₙₑₐᵣ = 1%', fontsize=35)
                color = "lightgreen"
            if stageNumber == "2nd stage":
                stage_curves = np.load(f"{resultPath}/stage2_curves.npy", allow_pickle=True).tolist()
                targetParams = parameterTypes[CPLaw]["small_yieldingParams"]
                fig.suptitle(f'2nd stage small yielding parameters calibration result | {CPLaw} model\nOptimizer: {optimizerName} | Searching space : {searchingSpace} | Allowed Δᵧ_ₗᵢₙₑₐᵣ = 0.5%', fontsize=35)
                color = "lightskyblue"
            if stageNumber == "3rd stage":
                stage_curves = np.load(f"{resultPath}/stage3_curves.npy", allow_pickle=True).tolist()
                targetParams = parameterTypes[CPLaw]["large_hardeningParams"]
                #fig.suptitle(f'3rd stage large hardening parameters calibration result | {CPLaw} model\nOptimizer: {optimizerName} | Searching space : {searchingSpace} | Allowed Δₕ_ₗᵢₙₑₐᵣ = 4.2%, Δₕ_ₙₒₙₗᵢₙₑₐᵣ = 6.2%', fontsize=35)
                fig.suptitle(f'3rd stage large hardening parameters calibration result | {CPLaw} model\nOptimizer: {optimizerName} | Searching space : {searchingSpace} | Allowed Δₕ_ₗᵢₙₑₐᵣ = 2%, Δₕ_ₙₒₙₗᵢₙₑₐᵣ = 3%', fontsize=35)
                color = "moccasin"
            if stageNumber == "4th stage":
                stage_curves = np.load(f"{resultPath}/stage4_curves.npy", allow_pickle=True).tolist()
                targetParams = parameterTypes[CPLaw]["small_hardeningParams"]
                fig.suptitle(f'4th stage small hardening parameters calibration result | {CPLaw} model\nOptimizer: {optimizerName} | Searching space : {searchingSpace} | Allowed Δₕ_ₗᵢₙₑₐᵣ = 4%, Δₕ_ₙₒₙₗᵢₙₑₐᵣ = 6%', fontsize=35)
                color = "lightcoral"


            tupleParams = stage_curves["parameters_tuple"]
     
            columnColors = [color]
            rowColors = []

            if stageNumber == "default parameters":
                iterationColumns = [f"{stageNumber}"]
            else:
                iterationColumns = [f"{stageNumber} result"]

            paramValues = []
            paramValues2D = []
            dictParams = dict(tupleParams)
            for param in dictParams:
                if param in targetParams:
                    paramValues.append(f"{round(dictParams[param], roundContinuousDecimals)} (target)")
                    rowColors.append(color)
                else:
                    paramValues.append(round(dictParams[param], roundContinuousDecimals))
                    rowColors.append("white")
            paramValues2D.append(paramValues)

            # st.write(paramValues2D)
            # transposing the matrix
            paramValues2D = np.array(paramValues2D).T

            size = 28

            for loading in loadings:
                iteration = 1
                pathTarget = f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_{curveType}.npy"
                target_Curve = np.load(pathTarget, allow_pickle=True).tolist()
                i = indexLoading[loading][0]
                j = indexLoading[loading][1]
                ax[i][j].plot(target_Curve["strain"], target_Curve["stress"] * convertUnit, color = "k", linewidth=3, alpha=1, label=f"Exp.\ncurve")
      
                ax[i][j].plot(stage_curves[curveType][loading]["strain"], stage_curves[curveType][loading]["stress"] * convertUnit,  linewidth=3, alpha=1, label=f"Sim.\ncurve")
                iteration += 1
                
                ax[i][j].set_xlim(right = 0.27)
                ax[i][j].set_ylim(top = 370)
                ax[i][j].tick_params(axis='x', labelsize= size)
                ax[i][j].tick_params(axis='y', labelsize= size)
                ax[i][j].set_ylabel('Stress, MPa', size= size)
                ax[i][j].set_xlabel('Strain, -', size= size)
                ax[i][j].set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
                ax[i][j].set_title(f"{loading} loading", size= 5/4 * size)
                legend = ax[i][j].legend(loc=4, frameon=False, fontsize= size - 2, ncol=2) #, shadow =True, framealpha=1)
                # plt.grid()
                legend.get_frame().set_linewidth(0.0)
            iTable = indexLoading["tableParam"][0]
            jTable = indexLoading["tableParam"][1]
            ax[iTable][jTable].axis('tight')
            ax[iTable][jTable].axis('off')
            table = ax[iTable][jTable].table(cellText=paramValues2D, 
                                    colLabels=iterationColumns, 
                                    rowLabels=parameterRows[CPLaw], 
                                    loc='upper center', 
                                    cellLoc='center', 
                                    colLoc="center",
                                    rowLoc="center",
                                    colWidths=[0.2 for x in iterationColumns], # Change this for column width
                                    colColours= columnColors, 
                                    rowColours= rowColors,
                                    fontsize=40)
            #ax[iTable][jTable].set_title(f"Parameter values", size= 5/4 * size)
            table.auto_set_column_width(col=iterationColumns)
            table.auto_set_font_size(True)
            table.set_fontsize(25)
            table.scale(3, 3) # Scaling the table

            st.pyplot(plt)

    with tab3:
        st.header('Plotting iteration MSE')
        loadings = ["linear_uniaxial_RD", 
                    "nonlinear_biaxial_RD", 
                    "nonlinear_biaxial_TD",     
                    "nonlinear_planestrain_RD",     
                    "nonlinear_planestrain_TD",     
                    "nonlinear_uniaxial_RD", 
                    "nonlinear_uniaxial_TD"]  
        # Extracting the results

        stage_CurvesList = np.load(f"{resultPath}/stage_CurvesList.npy", allow_pickle=True).tolist()
        
        ###########################
        startingIter = st.number_input("Starting iteration", min_value=-len(stage_CurvesList), max_value=len(stage_CurvesList), value=len(stage_CurvesList) - 6, key="MSE5")
        #startingIter = st.number_input("Starting iteration", value=23)

        endingIter = st.number_input("Ending iteration", min_value=-len(stage_CurvesList), max_value=len(stage_CurvesList), value=len(stage_CurvesList), key="MSE6")
        #endingIter = st.number_input("Ending iteration", value=26)
        
        #curveTypes = ("Total MSE", "Processed curves", "Interpolated curves")
        #curveType = st.radio("Please select the curve type", curveTypes)

        stage_CurvesListIter = stage_CurvesList[startingIter-1:endingIter]
        #stage_CurvesList = stage_CurvesList[-1:]
        hardeningLossActual = list([stageCurves["MSE"]["weighted_total_MSE"] for stageCurves in stage_CurvesListIter])
        hardeningLossPredicted = list([stageCurves["predicted_MSE"] for stageCurves in stage_CurvesListIter])
        iterationColumns =[]
 
  
        numberOfIterations = len(hardeningLossPredicted) 
        # for iteration in range(startingIter, endingIter):
        #     iterationColumns.append(int(iteration))
        for iteration in range(1, numberOfIterations + 1):
            iterationColumns.append(int(iteration))
        width = 0.4       # Width of the bar
        size = 20   # font size
        x_axis = np.arange(len(iterationColumns))
        PSO_error = [62.28392473874383, 44.479397383738383, 58.837483737737373, 42.28338383838838, 66.27388388383838, 45.24888484848484, 23.20483848488448]
        #GA_error = [42.412509787193535, 45.810850446470305, 31.678746652598473, 24.02204713416984, 64.66118552711472, 30.61382189223108, 22.28353410977453]
        fig, ax = plt.subplots(figsize=(8,5))
        #ax.bar(x_axis, hardeningLossPredicted, width, label=f'{optimizerName} predicted loss', color = "orange", align='center')
        ax.bar(x_axis, hardeningLossPredicted, width, label=f'{optimizerName} predicted loss', color = "orange", align='center')
        #ax.bar(x_axis + 0.2, hardeningLossActual, width, label = 'Actual loss', color="red")
        ax.set_title(f'Weighted hardening loss ({optimizerName} | {CPLaw})',size=size + 5)
        ax.set_ylabel('Loss', size=size+2)
        ax.set_xlabel("Iteration", size=size+2)
        ax.set_xticks(x_axis, iterationColumns, fontsize=size)
        ax.tick_params(axis='both', which='major', labelsize=size)
        ax.legend(fontsize=size-2)
        ax.set_ylim([0,40])
        #ax.set_ylim([0, 80])
        ax.tick_params(axis='both', which='major', labelsize=size)
        st.pyplot(plt)

        #################################################################
        startingIter = st.number_input("Starting iteration", min_value=-len(stage_CurvesList), max_value=len(stage_CurvesList), value=len(stage_CurvesList) - 9, key="MSE1")
        #startingIter = st.number_input("Starting iteration", value=23)

        endingIter = st.number_input("Ending iteration", min_value=-len(stage_CurvesList), max_value=len(stage_CurvesList), value=len(stage_CurvesList), key="MSE2")
        #endingIter = st.number_input("Ending iteration", value=26)
        
        #curveTypes = ("Total MSE", "Processed curves", "Interpolated curves")
        #curveType = st.radio("Please select the curve type", curveTypes)

        stage_CurvesListIter = stage_CurvesList[startingIter-1:endingIter]
        #stage_CurvesList = stage_CurvesList[-1:]
        hardeningLossActual = list([stageCurves["MSE"]["weighted_total_MSE"] for stageCurves in stage_CurvesListIter])
        hardeningLossPredicted = list([stageCurves["predicted_MSE"] for stageCurves in stage_CurvesListIter])
        #hardeningLossPredicted = [24.02527440321395, 27.191144810405113, 43.91997850916784, 42.412509787193535, 45.810850446470305, 31.678746652598473, 24.02204713416984, 64.66118552711472, 30.61382189223108, 22.28353410977453]
        #hardeningLossPredicted = [26.22717804043217, 18.5031735479146, 34.8363950236515, 29.551704490541887, 41.415524856338145, 33.455054377427885, 35.391435929280355, 19.139479031711144,37.529326867507905, 19.95558947959972]
        iterationColumns =[]
 
  
        numberOfIterations = len(hardeningLossPredicted) 
        # for iteration in range(startingIter, endingIter):
        #     iterationColumns.append(int(iteration))
        for iteration in range(1, numberOfIterations + 1):
            iterationColumns.append(int(iteration))
        width = 0.3       # Width of the bar
        size = 20   # font size
        x_axis = np.arange(len(iterationColumns))

        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(x_axis - 0.2, hardeningLossPredicted, width, label='ANN predicted loss', color = "orange", align='center')
        ax.bar(x_axis + 0.2, hardeningLossActual, width, label = 'Actual loss', color="red")
        ax.set_title(f'Weighted hardening loss ({optimizerName} | {CPLaw})',size=size + 5)
        ax.set_ylabel('Loss', size=size+2)
        ax.set_xlabel("Iteration", size=size+2)
        ax.set_xticks(x_axis, iterationColumns, fontsize=size)
        ax.tick_params(axis='both', which='major', labelsize=size)
        ax.legend(fontsize=size-2)
        #ax.set_ylim([0, 80])
        ax.tick_params(axis='both', which='major', labelsize=size)
        st.pyplot(plt)

        #################################################################
        weightsLoading = {
            "linear_uniaxial_RD": 0.33, 
            "nonlinear_biaxial_RD": 1, 
            "nonlinear_biaxial_TD": 1,     
            "nonlinear_planestrain_RD": 1,     
            "nonlinear_planestrain_TD": 1,     
            "nonlinear_uniaxial_RD": 1, 
            "nonlinear_uniaxial_TD": 1,
        }
        loadingsNewline = ["linear\nuniaxial\nRD", 
                    "nonlinear\nbiaxial\nRD", 
                    "nonlinear\nbiaxialn\nTD",     
                    "nonlinear\nplanestrain\nRD",     
                    "nonlinear\nplanestrain\nTD",     
                    "nonlinear\nuniaxial\nRD", 
                    "nonlinear\nuniaxial\nTD"]  
        iteration = st.number_input("MSE of iteration", value=-1)
        stage_CurvesListLoading = stage_CurvesList[iteration - 1]
        loadingMSE = []
        #st.write(stage_CurvesListLoading)
        for loading in loadings:
            #loadingMSE.append(stage_CurvesListLoading["MSE"][loading]["weighted_loading_MSE"])
            loadingMSE.append(stage_CurvesListLoading["MSE"][loading]["weighted_loading_MSE"] * weightsLoading[loading])
        width = 0.4       # Width of the bar
        size = 13   # font size

        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(loadingsNewline, loadingMSE, width, label='Actual loss', color = "red", align='center')
        
        ax.set_title(f'Weighted MSE of individual loadings ({optimizerName} | {CPLaw})',size=size + 7)
        
        ax.set_xlabel("Loading", size=size+2)
        ax.set_ylabel("Loss", size=size+2)
        ax.tick_params(axis='both', which='major', labelsize=size)
        ax.set_ylim([0,30])
        ax.legend(loc=1,prop={'size': size+2})
        #ax.set_ylim([0, 80])
        #ax.tick_params(axis='both', which='major', labelsize=size)
        st.pyplot(plt)
        
        ###############################################################################
        # Extracting the results

        stage_CurvesList = np.load(f"{resultPath}_yielding/stage_CurvesList.npy", allow_pickle=True).tolist()
        # stage_CurvesList = np.load(f"{resultPath}_yielding/stage_CurvesList.npy", allow_pickle=True).tolist()
        startingIterYielding = st.number_input("Starting iteration", min_value=-len(stage_CurvesList), max_value=len(stage_CurvesList), value=-2, key="MSE3")
        #startingIter = st.number_input("Starting iteration", value=23)

        endingIterYielding = st.number_input("Ending iteration", min_value=-len(stage_CurvesList), max_value=len(stage_CurvesList), value=-1, key="MSE4")

        stage_CurvesListIter = stage_CurvesList[startingIterYielding - 1:endingIterYielding]
        #stage_CurvesList = stage_CurvesList[-1:]
        yieldingLossANN = list([stageCurves["MSE"]["linear_uniaxial_RD"]["weighted_loading_MSE"] for stageCurves in stage_CurvesListIter])
        yieldingLossActual = list([stageCurves["predicted_MSE"] for stageCurves in stage_CurvesListIter])
        iterationColumns =[]
 

        numberOfIterations = len(yieldingLossANN) 
        # for iteration in range(startingIter, endingIter):
        #     iterationColumns.append(int(iteration))
        for iteration in range(1, numberOfIterations + 1):
            iterationColumns.append(int(iteration))
        width = 0.3       # Width of the bar
        size = 20   # font size

        fig, ax = plt.subplots(figsize=(8,5))

        x_axis = np.arange(len(iterationColumns))

        # Multi bar Chart

        ax.bar(x_axis - 0.2, yieldingLossANN, width=0.4, label = 'ANN predicted loss', color="orange")
        ax.bar(x_axis + 0.2, yieldingLossActual, width=0.4, label = 'Actual loss', color="red")
        ax.set_xticks(x_axis, iterationColumns, fontsize=size)
        ax.tick_params(axis='both', which='major', labelsize=size)
        ax.legend(fontsize=size-2)
        ax.set_title(f"Weighted yielding loss ({optimizerName} | {CPLaw})", fontsize=size+2)
        ax.set_ylabel("Loss", fontsize=size)
        ax.set_xlabel("Iterations", fontsize=size)
        st.pyplot(plt)   

    with tab4:
        st.header('Plot parameter analysis')
        loadings = ["linear_uniaxial_RD", 
                    "nonlinear_biaxial_RD", 
                    "nonlinear_biaxial_TD",     
                    "nonlinear_planestrain_RD",     
                    "nonlinear_planestrain_TD",     
                    "nonlinear_uniaxial_RD", 
                    "nonlinear_uniaxial_TD"]
        
        paramNames = {
            "PH": {
                "fitting_params": ["a", "gdot0", "h0", "n", "tau0", "tausat"],
                "interaction_coeffs": ["self", "coplanar", "collinear", "orthogonal", "glissile", "sessile"],
            },
            "DB": {
                "fitting_params": ["dipole", "islip", "omega", "p", "q", "tausol",],
                "microstructure_params": [ "Qc", "Qs", "v0", "rho_e", "rho_d"]
            }
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
                "dipole": "dα", 
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
            },
        }

        paramsUnit = {
            "PH": {
                "a": "", # Empty string means this parameter is dimensionless
                "gdot0": "s⁻¹", 
                "h0": "MPa", 
                "n": "", 
                "tau0": "MPa", 
                "tausat": "MPa",
                "self": "", 
                "coplanar": "", 
                "collinear": "", 
                "orthogonal": "", 
                "glissile": "", 
                "sessile": "", 
            },
            "DB": {
                "dipole": "b", 
                "islip": "", 
                "omega": "b³", 
                "p": "", 
                "q": "", 
                "tausol": "MPa",
                "Qs": "J",
                "Qc": "J",
                "v0": "m/s",
                "rho_e": "m⁻²",
                "rho_d": "m⁻²",   
            },
        }

        curveTypes = ["true", "process", "both"]

        if CPLaw == "PH":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Loading conditions")
                loading = st.radio("Please select the loading condition", loadings)
            with col2:
                st.markdown("Curve type")
                curveType = st.radio("Please select the curve type", curveTypes)
                if curveType == "both":
                    curveList = ["true","process"]
                elif curveType == "true":
                    curveList = ["true"]
                elif curveType == "process":
                    curveList = ["process"]
            col3, col4 =st.columns(2)
        
            with col3: 
                st.markdown("Fitting parameters")
                fitting_param = st.radio("Please select the fitting parameter", paramNames[CPLaw]["fitting_params"])
            with col4:
                st.markdown("Interaction coefficient")
                interaction_coeff = st.radio("Please select the interaction coefficient", paramNames[CPLaw]["interaction_coeffs"])
            
            #xlow = st.slider('low strain', 0.0, 0.2, 0.0, 0.001)
            #xhigh = st.slider('high strain', 0.0, 0.2, 0.2, 0.001)

            #ylow = st.slider('low stress', 0.0, 350.0, 0.0, 0.01)
            #yhigh = st.slider('low stress', 0.0, 350.0, 350.0, 0.01)

            true_data = np.load(f"parameter_analysis/{material}/{CPLaw}/results/analysis_trueCurves.npy", allow_pickle=True).tolist()
            process_data = np.load(f"parameter_analysis/{material}/{CPLaw}/results/analysis_processCurves.npy", allow_pickle=True).tolist()
            
            trueCurves_fitting = true_data[loading][fitting_param]
            trueCurves_interact = true_data[loading][interaction_coeff]
            
            processCurves_fitting = process_data[loading][fitting_param]
            processCurves_interact = process_data[loading][interaction_coeff]
            #st.write(trueCurves_fitting)

            # Plotting fitting param
            size = 10
            figure(figsize=(6, 4), dpi=80)
            formattedFitting = paramsFormatted[CPLaw][fitting_param]
            formattedInteraction = paramsFormatted[CPLaw][interaction_coeff]

            unitFitting = paramsUnit[CPLaw][fitting_param]
            unitInteraction = paramsUnit[CPLaw][interaction_coeff]
            if "true" in curveList:
                for (paramValue, strainstress) in trueCurves_fitting.items():
                    #st.write(strainstress)
                    strain = strainstress["strain"]
                    stress = strainstress["stress"] * convertUnit
                    plt.plot(strain, stress, label = f"True curve: {formattedFitting} = {paramValue} {unitFitting}")
            if "process" in curveList:
                for (paramValue, strainstress) in processCurves_fitting.items():
                    #st.write(strainstress)
                    strain = strainstress["strain"]
                    stress = strainstress["stress"] * convertUnit
                    plt.plot(strain, stress, label = f" Process curve: {formattedFitting} - {paramValue} {unitFitting}")
            plt.title(f"Varying values of fitting parameter \"{formattedFitting}\" ({material} - {CPLaw} law)", size=size + 2)
            #plt.xlim([xlow, xhigh])
            #plt.ylim([ylow, yhigh])
            plt.rc('xtick', labelsize=size - 2)    
            plt.rc('ytick', labelsize=size - 2)  
            plt.ylabel('Stress (MPa)', size=size)
            plt.xlabel("Strain (-)", size=size)
            plt.legend(loc=4, fontsize=size)
            st.pyplot(plt)

            # Plotting interaction coefficient
            size = 10
            figure(figsize=(6, 4), dpi=80)
            if "true" in curveList:
                for (paramValue, strainstress) in trueCurves_interact.items():
                    strain = strainstress["strain"]
                    stress = strainstress["stress"] * convertUnit
                    plt.plot(strain, stress, label = f" True curve: {formattedInteraction} - {paramValue} {unitInteraction}")
            if "process" in curveList:
                for (paramValue, strainstress) in processCurves_interact.items():
                    #st.write(strainstress)
                    strain = strainstress["strain"]
                    stress = strainstress["stress"] * convertUnit
                    plt.plot(strain, stress, label = f" Process curve: {formattedInteraction} - {paramValue} {unitInteraction}")
            plt.title(f"Varying values of interaction coefficient \"{formattedInteraction}\" ({material} - {CPLaw} law)", size=size + 2)
            #plt.xlim([xlow, xhigh])
            #plt.ylim([ylow, yhigh])
            plt.rc('xtick', labelsize=size - 2)    
            plt.rc('ytick', labelsize=size - 2)  
            plt.ylabel('Stress (MPa)', size=size)
            plt.xlabel("Strain (-)", size=size)
            plt.legend(loc=4, fontsize=size)
            st.pyplot(plt)

        if CPLaw == "DB":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Loading conditions")
                loading = st.radio("Please select the loading condition", loadings)
            with col2:
                st.markdown("Curve type")
                curveType = st.radio("Please select the curve type", curveTypes)
                if curveType == "both":
                    curveList = ["true","process"]
                elif curveType == "true":
                    curveList = ["true"]
                elif curveType == "process":
                    curveList = ["process"]
            col3, col4 =st.columns(2)
        
            with col3: 
                st.markdown("Fitting parameters")
                fitting_param = st.radio("Please select the fitting parameter", paramNames[CPLaw]["fitting_params"])
            with col4:
                st.markdown("Microstructure parameters")
                microstructure_param = st.radio("Please select the microstructure parameters", paramNames[CPLaw]["microstructure_params"])
        
            #xlow = st.slider('low strain', 0.0, 0.2, 0.0, 0.001)
            #xhigh = st.slider('high strain', 0.0, 0.2, 0.2, 0.001)

            #ylow = st.slider('low stress', 0.0, 350.0, 0.0, 0.01)
            #yhigh = st.slider('low stress', 0.0, 350.0, 350.0, 0.01)

            true_data = np.load(f"parameter_analysis/{material}/{CPLaw}/results/analysis_trueCurves.npy", allow_pickle=True).tolist()
            process_data = np.load(f"parameter_analysis/{material}/{CPLaw}/results/analysis_processCurves.npy", allow_pickle=True).tolist()
            
            trueCurves_fitting = true_data[loading][fitting_param]
            trueCurves_microstructure = true_data[loading][microstructure_param]
            
            processCurves_fitting = process_data[loading][fitting_param]
            processCurves_microstructure = process_data[loading][microstructure_param]

            #st.write(trueCurves_fitting)

            # Plotting fitting param
            size = 10
            figure(figsize=(6, 4), dpi=80)
            formattedFitting = paramsFormatted[CPLaw][fitting_param]
            formattedMicrostructure = paramsFormatted[CPLaw][microstructure_param]
        
            unitFitting = paramsUnit[CPLaw][fitting_param]
            unitMicrostructure = paramsUnit[CPLaw][microstructure_param]

            if "true" in curveList:
                for (paramValue, strainstress) in trueCurves_fitting.items():
                    #st.write(strainstress)
                    strain = strainstress["strain"]
                    stress = strainstress["stress"] * convertUnit
                    plt.plot(strain, stress, label = f" True curve: {formattedFitting} - {paramValue} {unitFitting}")
            if "process" in curveList:
                for (paramValue, strainstress) in processCurves_fitting.items():
                    #st.write(strainstress)
                    strain = strainstress["strain"]
                    stress = strainstress["stress"] * convertUnit
                    plt.plot(strain, stress, label = f" Process curve: {formattedFitting} - {paramValue} {unitFitting}")
            plt.title(f"Varying values of fitting parameter \"{formattedFitting}\" ({material} - {CPLaw} law)", size=size + 2)
            #plt.xlim([xlow, xhigh])
            #plt.ylim([ylow, yhigh])
            plt.rc('xtick', labelsize=size - 2)    
            plt.rc('ytick', labelsize=size - 2)  
            plt.ylabel('Stress (MPa)', size=size)
            plt.xlabel("Strain (-)", size=size)
            plt.legend(loc=4, fontsize=size)
            st.pyplot(plt)

            # Plotting microstructure param
            size = 10
            figure(figsize=(6, 4), dpi=80)
            if "true" in curveList:
                for (paramValue, strainstress) in trueCurves_microstructure.items():
                    strain = strainstress["strain"]
                    stress = strainstress["stress"] * convertUnit
                    plt.plot(strain, stress, label = f" True curve: {formattedMicrostructure} - {paramValue} {unitMicrostructure}")
            if "process" in curveList:
                for (paramValue, strainstress) in processCurves_microstructure.items():
                    #st.write(strainstress)
                    strain = strainstress["strain"]
                    stress = strainstress["stress"] * convertUnit
                    plt.plot(strain, stress, label = f" Process curve: {formattedMicrostructure} - {paramValue} {unitMicrostructure}")
            plt.title(f"Varying values of microstructure parameter \"{formattedMicrostructure}\" ({material} - {CPLaw} law)", size=size + 2)
            #plt.xlim([xlow, xhigh])
            #plt.ylim([ylow, yhigh])

            plt.rc('xtick', labelsize=size - 2)    
            plt.rc('ytick', labelsize=size - 2)  
            plt.ylabel('Stress (MPa)', size=size)
            plt.xlabel("Strain (-)", size=size)
            plt.legend(loc=4, fontsize=size)
            st.pyplot(plt)

    # with tab5:
    #     st.header('Convergence prediction')
    #     convergingParams = np.load(f"results/{material}/{CPLaw}/universal/initial_params.npy", allow_pickle=True)
    #     convergingParams = convergingParams.tolist()
    #     #st.write(convergingParams[0])
    #     nonconvergingParams = np.load(f"results/{material}/{CPLaw}/universal/nonconverging_params.npy", allow_pickle=True)
    #     nonconvergingParams = nonconvergingParams.tolist()
    #     #st.write(nonconvergingParams[0])

    #     general_param_info = getGeneralRanges(material, CPLaw, searchingSpace, roundContinuousDecimals)   
    #     if CPLaw == "PH":
    #         col1, col2, col3, col4 = st.columns(4)
    #         paramValues = {}
    #         params = list(general_param_info.keys())
    #         infos = list(general_param_info.values())
            
    #         with st.form(key="Convergence analysis"):
    #             with col1:
    #                 st.write("Fitting parameters")
    #             for i in range(0,4):
    #                 with col1:
    #                     paramValues[params[i]] = st.number_input(params[i], min_value=infos[i]["low"], max_value=infos[i]["high"], value=infos[i]["default"], step=infos[i]["step"], key=params[i])
                
    #             with col2:
    #                 st.write("Interaction coefficient")
    #             for i in range(4,7):
    #                 with col2:
    #                     paramValues[params[i]] = st.number_input(params[i], min_value=infos[i]["low"], max_value=infos[i]["high"], value=infos[i]["default"], step=infos[i]["step"], key=params[i])
                
    #             with col3:
    #                 st.write("Interaction coefficient")
    #             for i in range(7,10):
    #                 with col3:
    #                     paramValues[params[i]] = st.number_input(params[i], min_value=infos[i]["low"], max_value=infos[i]["high"], value=infos[i]["default"], step=infos[i]["step"], key=params[i])
                
    #             st.form_submit_button("Submit")
    #             # if submitted:
    #             #     st.write(f"The params are {paramValues}")
    #         with col4:
    #             st.write("Binary classification")
    #             st.write("We predict that this parameter will converge")
            # Forms can be declared using the 'with' syntax

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

def preprocessExperimentalTrue(path, excel=False):
    if not excel:
        df = pd.read_csv(path, delimiter = "\t")
    else:
        df = pd.read_excel(path, usecols=["exp_strain","exp_stress"], engine="openpyxl")
    trueStrain = df["exp_strain"].to_numpy()
    trueStress = df["exp_stress"].to_numpy()
    trueStrain = trueStrain[~np.isnan(trueStrain)]
    trueStress = trueStress[~np.isnan(trueStress)]
    return {"strain": trueStrain, "stress": trueStress}

def preprocessExperimentalFitted(path, excel=False):
    if not excel:
        df = pd.read_csv(path, delimiter = "\t")
    else:
        df = pd.read_excel(path, usecols=["fitted_strain","fitted_stress"], engine="openpyxl")
    fittedStrain = df["fitted_strain"].to_numpy()
    fittedStress = df["fitted_stress"].to_numpy()
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

def interpolatingStrain(average_initialStrain, exp_strain, stress, yieldingPoint, loading):
    if loading == "linear_uniaxial_RD":
        beforeYieldingIndex = getIndexBeforeStrainLevel(average_initialStrain, yieldingPoint) 
        interpolatedStrain = average_initialStrain[beforeYieldingIndex:]
        # Strain level is added to the interpolating strains
        interpolatedStrain = np.insert(interpolatedStrain, 1, yieldingPoint)   
        # print(exp_strain[-1])
        # time.sleep(30)
        if interpolatedStrain[-1] > exp_strain[-1]:
            indexOfInterpolatedStrainAfterLastExpStrain = getIndexAfterStrainLevel(interpolatedStrain, exp_strain[-1])
            interpolatedStrain = interpolatedStrain[:indexOfInterpolatedStrainAfterLastExpStrain+1]
    else: 
        reloadingIndex = turningStressPoints(stress)[1]
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
    if loading == "linear_uniaxial_RD":
        # Allows extrapolation
        interpolatingFunction = interp1d(strain, stress, fill_value='extrapolate')
        # Calculate the stress values at the interpolated strain points
        interpolatedStress = interpolatingFunction(interpolatedStrain)
        
    else:
        if len(turningStressPoints(stress)) != 0:
            reloadingIndex = turningStressPoints(stress)[1]
            strain = strain[reloadingIndex:]
            stress = stress[reloadingIndex:]
        # Allows extrapolation
        interpolatingFunction = interp1d(strain, stress, fill_value='extrapolate')
        # Calculate the stress values at the interpolated strain points
        interpolatedStress = interpolatingFunction(interpolatedStrain)
    #print(interpolatedStress)
    return interpolatedStress 

if __name__  == "__main__":
    import streamlit as st
    import numpy as np
    # from modules.preprocessing import *
    # from modules.stoploss import *
    # from modules.helper import *
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    import math 
    import copy
    import time

    from matplotlib.pyplot import figure
    import matplotlib.pyplot as plt
    import math
    main()

        # initial_flowCurves = np.load(f'GUI/results/{material}/{CPLaw}/universal/linear_uniaxial_RD/initial_flowCurves.npy', allow_pickle=True)
        # initial_flowCurves = initial_flowCurves.tolist()
        # initialSims = len(initial_flowCurves)
        # plt.figure(0)
        # for curve in initial_flowCurves.values():
        #     trueStress = curve[1] * convertUnit
        #     truePlasticStrain = curve[0] 
        #     plt.plot(truePlasticStrain, trueStress, c='orange', alpha=0.05)
        # plt.plot(truePlasticStrain, trueStress, label = f"Initial simulations x{initialSims}",c='orange', alpha=0.05)
        # plt.title(f"{initialSims} initial simulations ({material}/{CPLaw}/linear_uniaxial_RD)", fontsize=16)
        # plt.xlabel(xlabel = "Strain (-)", fontsize=16)
        # plt.ylabel(ylabel = "Stress (MPa)", fontsize=16)
        # plt.axvline(x = yieldingPoints[CPLaw]["linear_uniaxial_RD"], color = 'black', label = f'Yield stress at strain level = {yieldingPoints[CPLaw]["linear_uniaxial_RD"]}', alpha=0.4)
        # plt.legend(loc=2, prop={'size': 10})
        # # fig, ax = plotAllCurves(elasticStress, elasticStrain, trimmedStress, trimmedStrain, trueStress, truePlasticStrain, elasticCheck, plasticCheck, flowCheck, info)
        # st.pyplot(plt)

        # simStrains = list(map(lambda x: x[0], list(initial_flowCurves.values())))
        # average_strain = np.array(simStrains).mean(axis=0)
        # targetCurve = st.file_uploader("Please upload the target curve", type=["xlsx"])



        # general_param_info = getGeneralRanges(material, CPLaw, searchingSpace, roundContinuousDecimals)

        # param_info_no_round = param_info_no_round_func(general_param_info) # For GA discrete
        # param_info_no_step_dict = param_info_no_step_dict_func(param_info_no_round) # For GA continuous
        # param_info_no_step_tuple = param_info_no_step_tuple_func(param_info_no_round) # For BO discrete and continuous

        # # st.write(general_param_range)

        # if targetCurve is not None:
        #     # To read file as bytes:
        #     exp_curve = pd.read_excel(targetCurve)
        #     st.write("The target curve is")
        #     st.write(exp_curve)
        #     exp_stress = exp_curve.iloc[:,0] # Getting the experimental stress
        #     exp_strain = exp_curve.iloc[:,1] # Getting the experimental strain
        #     interpolated_strain = calculateInterpolatingStrainsLinear(simStrains, exp_strain, average_strain, yieldingPoints[CPLaw]["linear_uniaxial_RD"]) 
        #     exp_stress = interpolatedStressFunction(exp_stress, exp_strain, interpolated_strain).reshape(-1) * convertUnit
        #     exp_strain = interpolated_strain
            
        #     st.write("Start training the response surface (MLP)")
        #     X = np.array(list(initial_flowCurves.keys()))
        #     # Output layer of the size of the interpolated stresses
        #     y = np.array([interpolatedStressFunction(simStress, simStrain, exp_strain) * convertUnit for (simStrain, simStress) in initial_flowCurves.values()])
        #     inputSize = X.shape[1]
        #     outputSize = y.shape[1]
        #     hiddenSize1 = round((1/3) * (inputSize + outputSize))
        #     hiddenSize2 = round((2/3) * (inputSize + outputSize))
        #     regressor = MLPRegressor(hidden_layer_sizes=[hiddenSize1, hiddenSize2],activation='relu', alpha=0.001, solver='adam', max_iter=100000, shuffle=True)
        #     regressor = regressor.fit(X,y)
        #     st.write("Finish training the response surface (MLP)")

        #     info = {
        #         "param_info": general_param_info,
        #         "param_info_no_round": param_info_no_round,
        #         "param_info_no_step_tuple": param_info_no_step_tuple,
        #         "param_info_no_step_dict": param_info_no_step_dict,
        #         "exp_stress": exp_stress,
        #         "exp_strain": exp_strain,
        #         "regressor": regressor,
        #         "material": material,
        #         "CPLaw": CPLaw,
        #         "optimizerName": optimizerName,
        #         "convertUnit": convertUnit,
        #         "weightsYield": weightsYield,
        #         "weightsHardening": weightsHardening,
        #         "numberOfParams": numberOfParams,
        #         "searchingSpace": searchingSpace,   
        #         "roundContinuousDecimals": roundContinuousDecimals,
        #     }

        #     if optimizerName == "GA": 
        #         optimizer = GA(info)
        #     elif optimizerName == "BO":
        #         optimizer = BO(info)    
        #     #elif optimizerName == "PSO":
        #     #    optimizer = PSO(info) 
            
        #     y = np.array([interpolatedStressFunction(simStress, simStrain, exp_strain) * convertUnit for (simStrain, simStress) in initial_flowCurves.values()])
        #     # Obtaining the default hardening parameters
        #     targetYieldStress = exp_stress[1]
        #     zipParamsStress = list(zip(list(initial_flowCurves.keys()), y))
        #     shiftedToTargetYieldStress = list(map(lambda paramZipsimStress: (paramZipsimStress[0], paramZipsimStress[1] + (targetYieldStress - paramZipsimStress[1][1])), zipParamsStress))
        #     sortedClosestHardening = list(sorted(shiftedToTargetYieldStress, key=lambda pairs: fitness_hardening(exp_stress, pairs[1], exp_strain, weightsHardening["wh1"], weightsHardening["wh2"], weightsHardening["wh3"], weightsHardening["wh4"])))
        #     default_hardening_params = sortedClosestHardening[0][0]
        #     default_hardening_params = tupleOrListToDict(default_hardening_params, CPLaw)
        #     st.write("The default parameters are:")
        #     st.write(default_hardening_params)

        #     st.write(f"First stage optimization started by {optimizerName}")
        #     optimizer.InitializeFirstStageOptimizer(default_hardening_params)
        #     optimizer.FirstStageRun()
        #     partialResults = optimizer.FirstStageOutputResult()
        #     optimized_yielding_params = partialResults["solution_dict"]
        #     st.write("The partial result is: ")
        #     st.write(optimized_yielding_params)

        #     st.write(f"Second stage optimization started by {optimizerName}")
        #     optimizer.InitializeSecondStageOptimizer(optimized_yielding_params)
        #     optimizer.SecondStageRun()
        #     finalResults = optimizer.SecondStageOutputResult()
        #     st.write("The final result is: ")
        #     optimizedParams = finalResults["solution_dict"]
        #     st.write(optimizedParams)

        #     if CPLaw == "PH":
        #         fullSolution = np.array([optimizedParams['a'], optimizedParams['h0'], optimizedParams['tau0'], optimizedParams['tausat']])
        #     elif CPLaw == "DB":
        #         fullSolution = np.array([optimizedParams['dipole'], optimizedParams['islip'], optimizedParams['omega'], optimizedParams['p'], optimizedParams['q'], optimizedParams['tausol']])
        #     predicted_sim_stress = regressor.predict(fullSolution.reshape((1, numberOfParams))).reshape(-1)
            
        #     plt.figure(1)
        #     plt.plot(exp_strain, exp_stress, label = f"Target curve",c='red', alpha=0.5)
        #     if CPLaw == "PH":
        #         sat = get_sub("sat")
        #         titleParams = f"Final parameters: (a, h₀, \u03C4₀, \u03C4{sat})"
        #         parameters = (optimizedParams['a'], optimizedParams['h0'], optimizedParams['tau0'], optimizedParams['tausat'])
        #     elif CPLaw == "DB":
        #         slip = get_sub("slip")
        #         sol = get_sub("sol")
        #         parameters = (optimizedParams['dipole'], optimizedParams['islip'], optimizedParams['omega'], optimizedParams['p'], optimizedParams['q'], optimizedParams['tausol'])
        #         titleParams = f"Final parameters: (dᵅ, i{slip}, \u03C9, p, q, \u03C4{sol})"
            
        #     plt.plot([], [], ' ', label=titleParams)
        #     plt.plot(exp_strain, predicted_sim_stress, label = f"Simulated curve: {parameters}",c='blue', alpha=0.5)

        #     plt.title(f"Final simulation curve result", fontsize=16)
        #     plt.xlabel(xlabel = "Strain (-)", fontsize=14)
        #     plt.ylabel(ylabel = "Stress (MPa)", fontsize=14)
        #     plt.legend(loc=4, prop={'size': 10})
        #     # fig, ax = plotAllCurves(elasticStress, elasticStrain, trimmedStress, trimmedStrain, trueStress, truePlasticStrain, elasticCheck, plasticCheck, flowCheck, info)
        #     st.pyplot(plt)


# Run the program with the following command:
# python -m streamlit run GUI/GUI.py
# python3 -m streamlit run GUI/GUI.py


