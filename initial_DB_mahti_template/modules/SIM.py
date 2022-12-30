import os
import numpy as np
import random
import shutil
from .preprocessing import *
import time
from subprocess import Popen
from .helper import *
import copy

class SIM:
    def __init__(
        self,
        info=None
    ):
        self.info = info
        self.fileIndex = 1
        self.path2params = {} # loading/fileIndex -> param
        self.path2paramsAnalysis = {} #CPLaw/loading/parameter/fileIndex -> paramValue
        self.allowedNumberOfNonconvergences = 10
        

        self.analysis_processCurves = {}
        self.analysis_trueCurves = {}

        self.initial_params = None
        self.nonconvergingParams = set()
        self.initial_processCurves = {}
        self.initial_trueCurves = {}
        
        self.iteration_processCurves = {}
        self.iteration_trueCurves = {}

    ######################
    # Parameter analysis #
    ######################

    def run_parameter_analysis(self, baseParams, stepParams, numberOfAnalysisCurves, paramNames):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        loadings = self.info['loadings']
        #middle_index = int((numberOfAnalysisCurves - 1)/2)
        middle_fileindex = str(int((numberOfAnalysisCurves + 1)/2))
        print(middle_fileindex)
        print("Base parameters:")
        print(baseParams)
        numberOfJobsRequired = 7 * (numberOfAnalysisCurves - 1) * len(paramNames) + 7

        for loading in loadings:
            self.path2paramsAnalysis[loading] = {}
            simPathBase = f"parameter_analysis/{material}/{CPLaw}/{loading}/base"
            for fileIndex in os.listdir(simPathBase):
                shutil.rmtree(f"{simPathBase}/{fileIndex}")
         
            self.fileIndex = 1
            baseParamsCopy = copy.deepcopy(baseParams)
            fileIndex = str(self.fileIndex)
            self.make_new_analysis_job(baseParams, baseParams, loading, "base", "1", middle_fileindex)

            for paramName in paramNames:   
                self.path2paramsAnalysis[loading][paramName] = {}
                simPath = f"parameter_analysis/{material}/{CPLaw}/{loading}/{paramName}"
                
                for fileIndex in os.listdir(simPath):
                    shutil.rmtree(f"{simPath}/{fileIndex}")

                divide = int((numberOfAnalysisCurves - 1)/2)

                paramShift = [*range(-divide, divide + 1)]
                paramDelta = [stepParams[paramName] * i for i in paramShift]
                paramValue = [round(baseParams[paramName] + d, self.info["rounding"]) for d in paramDelta]
                print(f'{loading}: {paramName} - {paramValue}')
                self.fileIndex = 1
                for param in paramValue:
                    baseParamsCopy = copy.deepcopy(baseParams)
                    baseParamsCopy[paramName] = param
                    fileIndex = str(self.fileIndex)
                    self.fileIndex += 1
                    self.make_new_analysis_job(baseParams, baseParamsCopy, loading, paramName, fileIndex, middle_fileindex)
        
        self.submit_array_analysis_jobs(numberOfAnalysisCurves, numberOfJobsRequired, paramNames, middle_fileindex)
        self.save_analysis_outputs(baseParams, paramNames, middle_fileindex)
        print("Parameter analysis has been successfully completed and results were saved")


    def make_new_analysis_job(self, baseParams, params, loading, paramName, fileIndex, middle_fileindex):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        simPath = f"parameter_analysis/{material}/{CPLaw}/{loading}/{paramName}/{fileIndex}"
        if paramName != "base":
            if fileIndex != middle_fileindex:
                self.path2paramsAnalysis[loading][paramName][fileIndex] = params[paramName]
            else:
                self.path2paramsAnalysis[loading][paramName][fileIndex] = baseParams[paramName]
        
        if fileIndex != middle_fileindex:
            shutil.copytree(f"templates/{material}/{CPLaw}/{loading}", simPath)
            if self.info["CPLaw"] == "PH":
                self.edit_material_parameters_PH(params, simPath)
            if self.info["CPLaw"] == "DB":
                self.edit_material_parameters_DB(params, simPath)

    def submit_array_analysis_jobs(self, numberOfAnalysisCurves, numberOfJobsRequired, paramNames, middle_fileindex):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        projectPath = self.info['projectPath']
        loadings = self.info["loadings"]
        hyperqueue = self.info["hyperqueue"]
        server = self.info["server"]
        indices = [*range(1, numberOfAnalysisCurves + 1)]
        del indices[int(middle_fileindex) - 1]
        #print(indices)

        if hyperqueue == "no":
            with open("linux_slurm/array_file.txt", 'w') as filename:
                for loading in loadings:
                    filename.write(f"{projectPath}/parameter_analysis/{material}/{CPLaw}/{loading}/base/1\n")
                    for paramName in paramNames:
                        for index in indices:
                            filename.write(f"{projectPath}/parameter_analysis/{material}/{CPLaw}/{loading}/{paramName}/{index}\n")
            print("Number of jobs required: ", numberOfJobsRequired)
            print("Parameter analysis preprocessing stage starts")
            os.system(f"sbatch --wait --array=1-{numberOfJobsRequired} linux_slurm/{server}_array_pre.sh {material}")
            print("Parameter analysis preprocessing stage finished. The postprocessing stage starts")
            os.system(f"sbatch --wait --array=1-{numberOfJobsRequired} linux_slurm/{server}_array_post.sh {material}")
            print("Parameter analysis postprocessing stage finished")
        
        if hyperqueue == "yes":
            with open("linux_slurm/hyperqueue_file.txt", 'w') as filename:
                for loading in loadings:
                    filename.write(f"{projectPath}/parameter_analysis/{material}/{CPLaw}/{loading}/base/1 {material}\n")
                    for paramName in paramNames:
                        for index in indices:
                            filename.write(f"{projectPath}/parameter_analysis/{material}/{CPLaw}/{loading}/{paramName}/{index} {material}\n")
            
            print("Parameter analysis preprocessing stage starts")
            os.system(f"sbatch --wait linux_slurm/{server}_hyperqueue_pre.sh")
            print("Parameter analysis preprocessing stage finished. The postprocessing stage starts")
            os.system(f"sbatch --wait linux_slurm/{server}_hyperqueue_post.sh")
            print("Parameter analysis postprocessing stage finished")

    def save_analysis_outputs(self, baseParams, paramNames, middle_fileindex):
        loadings = self.info["loadings"]
        material = self.info["material"]
        CPLaw = self.info["CPLaw"]
        param_info = self.info["param_info"]
        self.analysis_processCurves['baseParams'] = baseParams
        self.analysis_trueCurves['baseParams'] = baseParams
        for loading in loadings:
            print(self.path2paramsAnalysis)
            simPath = f"parameter_analysis/{material}/{CPLaw}/{loading}/base/1"
            path2txt = f'{simPath}/postProc/{material}_tensionX.txt'
            if loading == "linear_uniaxial_RD":
                trueCurves = preprocessDAMASKTrue(path2txt)
                processCurves = preprocessDAMASKLinear(path2txt)
            else: 
                trueCurves = preprocessDAMASKTrue(path2txt)
                processCurves = preprocessDAMASKNonlinear(path2txt)
            self.analysis_trueCurves[loading] = {}
            self.analysis_trueCurves[loading]['base'] = {}
            self.analysis_trueCurves[loading]['base']['strain'] = trueCurves[0]
            self.analysis_trueCurves[loading]['base']['stress'] = trueCurves[1]

            self.analysis_processCurves[loading] = {}
            self.analysis_processCurves[loading]['base'] = {}
            self.analysis_processCurves[loading]['base']['strain'] = processCurves[0]
            self.analysis_processCurves[loading]['base']['stress'] = processCurves[1]
            
            for paramName in paramNames:
                self.analysis_trueCurves[loading][paramName] = {}
                self.analysis_processCurves[loading][paramName] = {}
                for (fileindex, paramValue) in self.path2paramsAnalysis[loading][paramName].items():
                    exponent = param_info[paramName]["exponent"]
                    if str(exponent) == "e0":
                        exponentSuffix = ""
                    else:
                        exponentSuffix = str(exponent)
                    paramValueString = str(paramValue) + exponentSuffix
                    if fileindex != middle_fileindex:
                        simPath = f"parameter_analysis/{material}/{CPLaw}/{loading}/{paramName}/{fileindex}"
                        path2txt = f'{simPath}/postProc/{material}_tensionX.txt'
                        if loading == "linear_uniaxial_RD":
                            trueCurves = preprocessDAMASKTrue(path2txt)
                            processCurves = preprocessDAMASKLinear(path2txt)
                        else: 
                            trueCurves = preprocessDAMASKTrue(path2txt)
                            processCurves = preprocessDAMASKNonlinear(path2txt)
                        self.analysis_trueCurves[loading][paramName][paramValueString] = {}
                        self.analysis_trueCurves[loading][paramName][paramValueString]['strain'] = trueCurves[0]
                        self.analysis_trueCurves[loading][paramName][paramValueString]['stress'] = trueCurves[1]

                        self.analysis_processCurves[loading][paramName][paramValueString] = {}
                        self.analysis_processCurves[loading][paramName][paramValueString]['strain'] = processCurves[0]
                        self.analysis_processCurves[loading][paramName][paramValueString]['stress'] = processCurves[1]
                    else: 
                        self.analysis_trueCurves[loading][paramName][paramValueString] = {}
                        self.analysis_trueCurves[loading][paramName][paramValueString]['strain'] = self.analysis_trueCurves[loading]['base']['strain']
                        self.analysis_trueCurves[loading][paramName][paramValueString]['stress'] = self.analysis_trueCurves[loading]['base']['stress']

                        self.analysis_processCurves[loading][paramName][paramValueString] = {}
                        self.analysis_processCurves[loading][paramName][paramValueString]['strain'] = self.analysis_trueCurves[loading]['base']['strain']
                        self.analysis_processCurves[loading][paramName][paramValueString]['stress'] = self.analysis_trueCurves[loading]['base']['stress']
        np.save(f"parameter_analysis/{material}/{CPLaw}/results/analysis_trueCurves.npy", self.analysis_trueCurves) 
        np.save(f"parameter_analysis/{material}/{CPLaw}/results/analysis_processCurves.npy", self.analysis_processCurves)

    #######################
    # INITIAL SIMULATIONS #
    #######################

    def run_initial_simulations(self, tupleParams = None):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        loadings = self.info['loadings']
        if self.info["method"] == "auto":
            n_params = self.get_grid(self.info["initialSims"])
            np.save(f"results/{material}/{CPLaw}/universal/initial_params.npy", n_params)
            np.save(f"manualParams/{material}/{CPLaw}/initial_params.npy", n_params)
            print("\nInitial parameters generated")
            time.sleep(30)
        elif self.info["method"] == "manual":
            n_params = tupleParams
            self.initial_params = n_params
            np.save(f"results/{material}/{CPLaw}/universal/initial_params.npy", self.initial_params)

        for loading in loadings:
            self.path2params[loading] = {}
            simPath = f"simulations/{material}/{CPLaw}/universal/{loading}"
            for fileIndex in os.listdir(simPath):
                shutil.rmtree(f"{simPath}/{fileIndex}")
            self.fileIndex = 0
            for params in n_params:
                self.fileIndex += 1
                fileIndex = str(self.fileIndex)
                self.make_new_initial_job(params, loading, fileIndex)
        self.submit_initial_jobs()
        self.save_initial_outputs()

    def make_new_initial_job(self, params, loading, fileIndex):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        simPath = f"simulations/{material}/{CPLaw}/universal/{loading}/{fileIndex}"
        shutil.copytree(f"templates/{material}/{CPLaw}/{loading}", simPath) 
        self.path2params[loading][fileIndex] = params
        #self.initial_params[fileIndex - 1] = params
        #np.save(f"results/{material}/{CPLaw}/universal/initial_params.npy", self.initial_params)
        if self.info["CPLaw"] == "PH":
            self.edit_material_parameters_PH(params, simPath)
        if self.info["CPLaw"] == "DB":
            self.edit_material_parameters_DB(params, simPath)

    def submit_initial_jobs(self):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        projectPath = self.info['projectPath']
        loadings = self.info["loadings"]
        hyperqueue = self.info["hyperqueue"]
        server = self.info["server"]
        #rangeIndices = self.jobIndices([1, self.fileIndex], "range")
        #time.sleep(20)
        if hyperqueue == "no":
            with open("linux_slurm/array_file.txt", 'w') as filename:
                for loading in loadings:
                    for index in range(1, self.fileIndex + 1):
                        filename.write(f"{projectPath}/simulations/{material}/{CPLaw}/universal/{loading}/{index}\n")
            print("Initial simulation preprocessing stage starts")
            numberOfJobsRequired = self.fileIndex * 7 
            print("Number of jobs required:", numberOfJobsRequired)
            os.system(f"sbatch --wait --array=1-{numberOfJobsRequired} linux_slurm/{server}_array_pre.sh {material}")
        
        elif hyperqueue == "yes":
            with open("linux_slurm/hyperqueue_file.txt", 'w') as filename:
                for loading in loadings:
                    for index in range(1, self.fileIndex + 1):
                        filename.write(f"{projectPath}/simulations/{material}/{CPLaw}/universal/{loading}/{index} {material}\n")
            print("Initial simulation preprocessing stage starts")
            numberOfJobsRequired = self.fileIndex * 7 
            print("Number of jobs required:", numberOfJobsRequired)
            os.system(f"sbatch --wait linux_slurm/{server}_hyperqueue_pre.sh")

        while True:
            nonconvergings = []
            for loading in loadings:
                simPath = f"simulations/{material}/{CPLaw}/universal/{loading}"
                nonconvergings.extend(self.nonconvergingSims(simPath, loading))
            nonconvergings = list(set(nonconvergings))
            # Saving nonconvergin params
            for index in nonconvergings:
                self.nonconvergingParams.add(tuple(self.path2params["linear_uniaxial_RD"][str(index)].items()))
            np.save(f'results/{material}/{CPLaw}/universal/nonconverging_params.npy', self.nonconvergingParams)
            
            # If all loadings successfully converge then this should be empty list
            if len(nonconvergings) == 0:
                break

            # If not, printing the index file and regenerating jobs
            print("The non-converging sims among all loadings are: ")
            print(nonconvergings)
            n_params = self.get_grid(len(nonconvergings))
            time.sleep(10)
            self.regenerate(nonconvergings, n_params)

            # Saving new params
            np.save(f'results/{material}/{CPLaw}/universal/initial_params.npy', list(self.path2params["linear_uniaxial_RD"].values()))

            #someIndices = self.jobIndices(nonconvergings, "some")
            if hyperqueue == "no":
                with open("linux_slurm/array_file.txt", 'w') as filename:
                    for loading in loadings:
                        for index in nonconvergings:
                            filename.write(f"{projectPath}/simulations/{material}/{CPLaw}/universal/{loading}/{index}\n")
                numberOfJobsRequired = len(nonconvergings) * 7
                print("Number of jobs required:", numberOfJobsRequired)
                print("Rerunning initial simulation preprocessing stage for nonconverging parameters")
                os.system(f"sbatch --wait --array=1-{numberOfJobsRequired} linux_slurm/{server}_array_pre.sh {material}")

            elif hyperqueue == "yes":
                with open("linux_slurm/hyperqueue_file.txt", 'w') as filename:
                    for loading in loadings:
                        for index in nonconvergings:
                            filename.write(f"{projectPath}/simulations/{material}/{CPLaw}/universal/{loading}/{index} {material}\n")
                numberOfJobsRequired = len(nonconvergings) * 7
                print("Number of jobs required:", numberOfJobsRequired)
                print("Rerunning initial simulation preprocessing stage for nonconverging parameters")
                os.system(f"sbatch --wait linux_slurm/{server}_hyperqueue_pre.sh")

        print("All initial simulations of all loadings successfully converge")
        print("Initial simulations preprocessing stage finished. The postprocessing stage starts") 

        if hyperqueue == "no":
            with open("linux_slurm/array_file.txt", 'w') as filename:
                for loading in loadings:
                    for index in range(1, self.fileIndex + 1):
                        filename.write(f"{projectPath}/simulations/{material}/{CPLaw}/universal/{loading}/{index}\n")

            numberOfJobsRequired = self.fileIndex * 7
            print("Number of jobs required:", numberOfJobsRequired) 
            os.system(f"sbatch --wait --array=1-{numberOfJobsRequired} linux_slurm/{server}_array_post.sh {material}")
            print("Initial simulation postprocessing stage finished")
        
        elif hyperqueue == "yes":
            with open("linux_slurm/hyperqueue_file.txt", 'w') as filename:
                for loading in loadings:
                    for index in range(1, self.fileIndex + 1):
                        filename.write(f"{projectPath}/simulations/{material}/{CPLaw}/universal/{loading}/{index} {material}\n")
            numberOfJobsRequired = self.fileIndex * 7
            print("Number of jobs required:", numberOfJobsRequired) 
            os.system(f"sbatch --wait linux_slurm/{server}_hyperqueue_post.sh")
            print("Initial simulation postprocessing stage finished")
            
    def nonconvergingSims(self, simPath, loading):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        tensionXload = pd.read_table(f"templates/{material}/{CPLaw}/{loading}/tensionX.load",header=None)
        numberOfRows = (tensionXload.shape)[0]
        nonconverging = []
        totalIncrements = 0
        for i in range(numberOfRows):
            splitRow = tensionXload.iloc[i,0].split(" ")
            for j in range(len(splitRow)):
                if splitRow[j] == "incs":
                    totalIncrements += int(splitRow[j + 1])
        
        for index in range(1, self.fileIndex + 1):
            sta_path = f"{simPath}/{index}/{material}_tensionX.sta"
            if os.stat(sta_path).st_size == 0:
                nonconverging.append(index)
            else:
                with open(sta_path) as f:
                    lines = f.readlines() 
                    if len(lines) == 1: 
                        nonconverging.append(index)
                    else:
                        lastLine = lines[-1]
                        splitLine = lastLine.split(" ")
                        for num in splitLine:
                            if self.isfloat(num):
                                lastIteration = int(num)
                                break
                        if lastIteration != totalIncrements:
                            nonconverging.append(index)
        return nonconverging

    def regenerate(self, nonconverging, n_params):
        """
        regenerate the nonconverging params
        """
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        loadings = self.info["loadings"]
        for loading in loadings:
            for i in range(len(nonconverging)):
                index = str(nonconverging[i])
                simPath = f"simulations/{material}/{CPLaw}/universal/{loading}/{index}"
                params = n_params[i]
                if os.path.exists(simPath) and os.path.isdir(simPath):
                    shutil.rmtree(simPath)
                self.make_new_initial_job(params, loading, index)

    def save_initial_outputs(self):
        loadings = self.info["loadings"]
        material = self.info["material"]
        CPLaw = self.info["CPLaw"]
        np.save(f'results/{material}/{CPLaw}/universal/initial_params.npy', list(self.path2params["linear_uniaxial_RD"].values()))
        for loading in loadings:
            self.initial_trueCurves[loading] = {}
            self.initial_processCurves[loading] = {}
            #self.initial_trueCurves[tuple(params.items())] = {}
            #self.initial_processCurves[tuple(params.items())] = {}
            for (index, params) in self.path2params[loading].items():
                simPath = f"simulations/{material}/{CPLaw}/universal/{loading}/{index}"
                path2txt = f'{simPath}/postProc/{material}_tensionX.txt'
                if loading == "linear_uniaxial_RD":
                    processCurves = preprocessDAMASKLinear(path2txt)
                    trueCurves = preprocessDAMASKTrue(path2txt)
                else: 
                    processCurves = preprocessDAMASKNonlinear(path2txt)
                    trueCurves = preprocessDAMASKTrue(path2txt)
                self.initial_processCurves[loading][tuple(params.items())] = processCurves
                self.initial_trueCurves[loading][tuple(params.items())] = trueCurves
                #self.initial_processCurves[tuple(params.items())][loading] = processCurves
                #self.initial_trueCurves[tuple(params.items())][loading] = trueCurves
        resultPath = f"results/{material}/{CPLaw}/universal"
        np.save(f"{resultPath}/initial_processCurves.npy", self.initial_processCurves)
        np.save(f"{resultPath}/initial_trueCurves.npy", self.initial_trueCurves) 

    #########################
    # ITERATION SIMULATIONS #
    #########################

    def run_iteration_simulations(self, tupleParams):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        loadings = self.info['loadings']
        optimizerName = self.info["optimizerName"]
        curveIndex = self.info["curveIndex"]

        self.fileIndex += 1
        fileIndex = str(self.fileIndex)
        
        for loading in loadings:
            self.path2params[loading] = {}
            simPath = f"simulations/{material}/{CPLaw}/{optimizerName}{curveIndex}/{loading}/{fileIndex}"
            if os.path.exists(simPath) and os.path.isdir(simPath):
                shutil.rmtree(simPath)
            self.make_new_iteration_job(tupleParams, loading, fileIndex)

        converging = self.submit_array_iteration_jobs()
        if converging: 
            self.save_single_output(tupleParams)
        return converging

    def make_new_iteration_job(self, params, loading, index):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        optimizerName = self.info["optimizerName"]
        curveIndex = self.info["curveIndex"]
        simPath = f"simulations/{material}/{CPLaw}/{optimizerName}{curveIndex}/{loading}/{index}"
        shutil.copytree(f"./templates/{material}/{CPLaw}/{loading}", simPath) 
        self.path2params[loading][index] = params
        if self.info["CPLaw"] == "PH":
            self.edit_material_parameters_PH(params, simPath)
        if self.info["CPLaw"] == "DB":
            self.edit_material_parameters_DB(params, simPath)

    def submit_array_iteration_jobs(self):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        projectPath = self.info['projectPath']
        loadings = self.info["loadings"]
        optimizerName = self.info["optimizerName"]
        curveIndex = self.info["curveIndex"]
        preprocessCommands = []
        postprocessCommands = []

        singleIndex = self.fileIndex
        for loading in loadings:
            fullpath = f"{projectPath}/simulations/{material}/{CPLaw}/{optimizerName}{curveIndex}/{loading}"
            preprocessCommand = f'sh linux_slurm/array_runsimpre.sh {singleIndex} {fullpath} {material}'
            preprocessCommands.append(preprocessCommand)
            postprocessCommand = f'sh linux_slurm/array_runsimpost.sh {singleIndex} {fullpath} {material}'
            postprocessCommands.append(postprocessCommand)
        # run commands in parallel
        processes = [Popen(command, shell=True) for command in preprocessCommands]
        _ = [p.wait() for p in processes]
        
        nonconvergings = []
        for loading in loadings:
            simPath = f"simulations/{material}/{CPLaw}/{optimizerName}{curveIndex}/{loading}"
            nonconvergings.append(self.nonconvergingSims(simPath, loading))
        nonconvergings = list(set(nonconvergings))
        if len(nonconvergings) == 0:
            processes = [Popen(command, shell=True) for command in postprocessCommands]
            _ = [p.wait() for p in processes]
            return True
        else: 
            return False

    def save_iteration_outputs(self, params):
        loadings = self.info["loadings"]
        material = self.info["material"]
        CPLaw = self.info["CPLaw"]
        self.initial_params
        np.save(f'results/{material}/{CPLaw}/universal/initial_params.npy', list(self.path2params["linear_uniaxial_RD"].values()))
        for loading in loadings:
            self.initial_trueCurves[loading] = {}
            self.initial_processCurves[loading] = {}
            resultPath = f"results/{material}/{CPLaw}/universal/{loading}"
            for (index, params) in self.path2params[loading].items():
                simPath = f"simulations/{material}/{CPLaw}/universal/{loading}/{index}"
                path2txt = f'{simPath}/postProc/{material}_tensionX.txt'
                if loading == "linear_uniaxial_RD":
                    processCurves = preprocessDAMASKLinear(path2txt)
                    trueCurves = preprocessDAMASKTrue(path2txt)
                else: 
                    processCurves = preprocessDAMASKNonlinear(path2txt)
                    trueCurves = preprocessDAMASKTrue(path2txt)
                self.initial_processCurves[loading][params] = processCurves
                self.initial_trueCurves[loading][params] = trueCurves
            np.save(f"{resultPath}/initial_processCurves.npy", self.initial_processCurves[loading])
            np.save(f"{resultPath}/initial_trueCurves.npy", self.initial_trueCurves[loading]) 

    #####################
    # Helper functions #
    ####################
    def checkReplace(self, paramName, param_dict):
        param_info = self.info["param_info"]
        exponent = param_info[paramName]["exponent"]
        if str(exponent) == "e0":
           exponentSuffix = ""
        else:
            exponentSuffix = str(exponent)
        if param_info[paramName]["optimized_target"] == True:
            return str(param_dict[paramName]) + exponentSuffix
        else: 
            return str(param_info[paramName]["default"]) + exponentSuffix

    def edit_material_parameters_PH(self, param_dict, job_path):
        path = f'{job_path}/material.config'
        with open(path) as f:
            lines = f.readlines()
        for i in range(100):
            # Fitting parameters
            if lines[i].startswith("gdot0_slip"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("gdot0", param_dict))
            if lines[i].startswith("n_slip"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("n", param_dict))
            if lines[i].startswith("a_slip"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("a", param_dict))
            if lines[i].startswith("h0_slipslip"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("h0", param_dict))
            if lines[i].startswith("tau0_slip"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("tau0", param_dict))
            if lines[i].startswith("tausat_slip"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("tausat", param_dict))  
            # Interaction coefficients
            if lines[i].startswith("interaction_slipslip"):
                lines[i] = self.replaceInteractionCoeffs(lines[i], param_dict)   
        with open(f'{job_path}/material.config', 'w') as f:
            f.writelines(lines)

    def edit_material_parameters_DB(self, param_dict, job_path):
        path = f'{job_path}/material.config'
        with open(path) as f:
            lines = f.readlines()
        for i in range(100):
            # Fitting parameters
            if lines[i].startswith("Cedgedipmindistance"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("dipole", param_dict))
            if lines[i].startswith("CLambdaSlip"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("islip", param_dict))
            if lines[i].startswith("Catomicvolume"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("omega", param_dict))
            if lines[i].startswith("p_slip"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("p", param_dict))    
            if lines[i].startswith("q_slip"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("q", param_dict)) 
            if lines[i].startswith("SolidSolutionStrength"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("tausol", param_dict))  
            # Microstructure parameters
            if lines[i].startswith("Qedge"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("Qs", param_dict))
            if lines[i].startswith("Qsd"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("Qc", param_dict))
            if lines[i].startswith("v0"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("v0", param_dict))
            if lines[i].startswith("rhoedge0"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("rho_e", param_dict))    
            if lines[i].startswith("rhoedgedip0"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("rho_d", param_dict))         
            # Interaction coefficients
            if lines[i].startswith("interaction_slipslip"):
                lines[i] = self.replaceInteractionCoeffs(lines[i], param_dict)
        with open(f'{job_path}/material.config', 'w') as f:
            f.writelines(lines)

    def jobIndices(self, indices, type):
        if type == "range":
            indicesString = f"{indices[0]}-{indices[1]}"
        if type == "some":
            indicesString = ','.join(str(e) for e in indices)
        return indicesString

    def get_grid(self, numberOfSims):
        points = []
        searchingSpace = self.info["searchingSpace"]
        np.random.seed(20)
        param_info = self.info["param_info"]
        #print(param_info)

        if searchingSpace == "continuous":  
            linspaceValues = {}
            for param in param_info:
                linspaceValues[param] = np.linspace(param_info[param]["low"], param_info[param]["high"], num = self.info["initialSims"])
                linspaceValues[param] = linspaceValues[param].tolist()   
            for _ in range(numberOfSims):
                while True:
                    candidateParam = {}
                    for param in linspaceValues:
                        random.shuffle(linspaceValues[param])
                        if param_info[param]["optimized_target"] == True:
                            candidateParam[param] = round(linspaceValues[param].pop(), self.info["roundContinuousDecimals"])
                    if candidateParam not in points:
                        break
                points.append(candidateParam)
        
        if searchingSpace == "discrete":  
            for _ in range(numberOfSims):
                while True:    
                    candidateParam = {}
                    for parameter in self.info['param_info']:
                        if param_info[param]["optimized_target"] == True:
                            candidateParam[parameter] = self.discrete_param(self.info['param_info'][parameter]['low'], 
                                                                            self.info['param_info'][parameter]['high'], 
                                                                            self.info['param_info'][parameter]['step'], 
                                                                            self.info['param_info'][parameter]['round'])
                    if candidateParam not in points:
                        break
                points.append(candidateParam)
        return points

    def discrete_param(self, low, high, step, roundDecimals):
        spaces = int((high - low) / step)
        return round(random.randint(0, spaces) * step + low, roundDecimals) 

    def continuous_param(self, low, high, roundContinuousDecimals):
        return round(random.uniform(low, high), roundContinuousDecimals)

    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    def replaceAllNumbersInLine(self, line, num):
        splitLine = line.split(" ")  
        #print(splitLine)
        for i in range(len(splitLine)):
            if splitLine[i].endswith("\n"):
                if self.isfloat(splitLine[i][:-1]): 
                    splitLine[i] = str(num) + "\n"
            elif self.isfloat(splitLine[i]):
                splitLine[i] = str(num)

        lineRebuilt = ""
        for word in splitLine:
            lineRebuilt += word + " "
        lineRebuilt = lineRebuilt[:-1]
        #print(lineRebuilt)
        return lineRebuilt

    def replaceInteractionCoeffs(self, line, param_dict):
        coefficients = {0: "self", 1: "coplanar", 2: "collinear", 3: "orthogonal", 4: "glissile", 5: "sessile"}
        splitLine = line.split(" ") 
        #print(splitLine) 
        counter = 0
        for i in range(len(splitLine)):
            if splitLine[i].endswith("\n"):
                if self.isfloat(splitLine[i][:-1]):     
                    splitLine[i] = str(self.checkReplace(coefficients[counter], param_dict)) + "\n"
                    counter += 1
            elif self.isfloat(splitLine[i]):
                splitLine[i] = str(self.checkReplace(coefficients[counter], param_dict))
                counter += 1

        lineRebuilt = ""
        for word in splitLine:
            lineRebuilt += word + " "
        lineRebuilt = lineRebuilt[:-1]
        #print(lineRebuilt)
        return lineRebuilt

