import os
import numpy as np
import random
import shutil
from .preprocessing import *
import time
import subprocess
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
      
        self.resultPath = None
        self.simulationPath = None

        self.initial_params = None
        self.nonconvergingParams = set()

        self.initial_processCurves = {}
        self.initial_trueCurves = {}
        
        self.iteration_processCurves = {}
        self.iteration_trueCurves = {}

    #######################
    # INITIAL SIMULATIONS #
    #######################

    def run_initial_simulations(self, dictParams = None):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        loadings = self.info['loadings']
        self.simulationPath = f"simulations/{material}/{CPLaw}/universal"
        self.resultPath = f"results/{material}/{CPLaw}/universal"

        if self.info["method"] == "auto":
            self.initial_params = self.latin_hypercube_sampling(self.info["initialSims"])
            for loading in loadings:
                np.save(f"{self.resultPath}/initial_params.npy", self.initial_params)
            np.save(f"manualParams/{material}/{CPLaw}/initial_params.npy", self.initial_params)
            print("\nInitial parameters generated")
            time.sleep(30)
        elif self.info["method"] == "manual":
            self.initial_params = dictParams
            np.save(f"{self.resultPath}/initial_params.npy", self.initial_params)

        for loading in loadings:
            self.path2params[loading] = {}
            simLoadingPath = f"{self.simulationPath}/{loading}"
            for fileIndex in os.listdir(simLoadingPath):
                shutil.rmtree(f"{simLoadingPath}/{fileIndex}")
            self.fileIndex = 0
            for params in self.initial_params:
                self.fileIndex += 1
                fileIndex = str(self.fileIndex)
                self.make_new_initial_job(params, loading, fileIndex)
        self.submit_initial_jobs()
        self.save_initial_outputs()

    def make_new_initial_job(self, params, loading, fileIndex):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        simLoadingIndexPath = f"{self.simulationPath}/{loading}/{fileIndex}"
        templateLoadingPath = f"templates/{material}/{CPLaw}/{loading}"
        shutil.copytree(templateLoadingPath, simLoadingIndexPath) 
        self.path2params[loading][fileIndex] = params
        if self.info["CPLaw"] == "PH":
            self.edit_material_parameters_PH(params, simLoadingIndexPath)
        if self.info["CPLaw"] == "DB":
            self.edit_material_parameters_DB(params, simLoadingIndexPath)

    def submit_initial_jobs(self):
        material = self.info["material"]
        projectPath = self.info['projectPath']
        loadings = self.info["loadings"]
        server = self.info["server"]
        exampleLoading = self.info["exampleLoading"]

        with open("linux_slurm/array_initial_file.txt", 'w') as filename:
            for loading in loadings:
                for index in range(1, self.fileIndex + 1):
                    filename.write(f"{projectPath}/{self.simulationPath}/{loading}/{index}\n")
        print("Initial simulation preprocessing stage starts")
        numberOfJobsRequired = self.fileIndex * len(loadings) 
        print("Number of jobs required:", numberOfJobsRequired)
        subprocess.run(f"sbatch --wait --array=1-{numberOfJobsRequired} linux_slurm/{server}_array_pre.sh {material} initial", shell=True)
        
        while True:
            nonconvergings = []
            for loading in loadings:
                nonconvergings.extend(self.nonconvergingSims(loading))
            nonconvergings = list(set(nonconvergings))
            # Saving nonconvergin params
            for index in nonconvergings:
                self.nonconvergingParams.add(tuple(self.path2params[exampleLoading][str(index)].items()))
            np.save(f'{self.resultPath}/nonconverging_params.npy', self.nonconvergingParams)
            
            # If all loadings successfully converge then this should be empty list
            if len(nonconvergings) == 0:
                break

            # If not, printing the index file and regenerating jobs
            print("The non-converging sims among all loadings are: ")
            print(nonconvergings)
            new_params = self.latin_hypercube_sampling(len(nonconvergings))
            time.sleep(10)
            self.regenerate(nonconvergings, new_params)

            # Saving new params and nonconverging params
            np.save(f'{self.resultPath}/initial_params.npy', list(self.path2params[exampleLoading].values()))

            #someIndices = self.jobIndices(nonconvergings, "some")

            with open("linux_slurm/array_initial_file.txt", 'w') as filename:
                for loading in loadings:
                    for index in nonconvergings:
                        filename.write(f"{projectPath}/{self.simulationPath}/{loading}/{index}\n")
            numberOfJobsRequired = len(nonconvergings) * len(loadings)
            print("Number of jobs required:", numberOfJobsRequired)
            print("Rerunning initial simulation preprocessing stage for nonconverging parameters")
            subprocess.run(f"sbatch --wait --array=1-{numberOfJobsRequired} linux_slurm/{server}_array_pre.sh {material} initial", shell=True)

        print("All initial simulations of all loadings successfully converge")
        print("Initial simulations preprocessing stage finished. The postprocessing stage starts") 

        with open("linux_slurm/array_initial_file.txt", 'w') as filename:
            for loading in loadings:
                for index in range(1, self.fileIndex + 1):
                    filename.write(f"{projectPath}/{self.simulationPath}/{loading}/{index}\n")

        numberOfJobsRequired = self.fileIndex * len(loadings)
        print("Number of jobs required:", numberOfJobsRequired) 
        subprocess.run(f"sbatch --wait --array=1-{numberOfJobsRequired} linux_slurm/{server}_array_post.sh {material} initial", shell=True)
        print("Initial simulation postprocessing stage finished")
        

    def nonconvergingSims(self, loading):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        tensionXload = pd.read_table(f"templates/{material}/{CPLaw}/{loading}/tensionX.load",header=None)
        numberOfRows = (tensionXload.shape)[0]
        nonconverging = []
        totalIncrements = 0
        simLoadingPath = f"{self.simulationPath}/{loading}"
        for i in range(numberOfRows):
            splitRow = tensionXload.iloc[i,0].split(" ")
            for j in range(len(splitRow)):
                if splitRow[j] == "incs":
                    totalIncrements += int(splitRow[j + 1])
        
        for index in range(1, self.fileIndex + 1):
            sta_path = f"{simLoadingPath}/{index}/{material}_tensionX.sta"
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

    def nonconvergingSimsIteration(self, simPath, loading):
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
        
        sta_path = f"{simPath}/{self.fileIndex}/{material}_tensionX.sta"
        if os.stat(sta_path).st_size == 0:
            nonconverging.append(self.fileIndex)
        else:
            with open(sta_path) as f:
                lines = f.readlines() 
                if len(lines) == 1: 
                    nonconverging.append(self.fileIndex)
                else:
                    lastLine = lines[-1]
                    splitLine = lastLine.split(" ")
                    for num in splitLine:
                        if self.isfloat(num):
                            lastIteration = int(num)
                            break
                    if lastIteration != totalIncrements:
                        nonconverging.append(self.fileIndex)
        return nonconverging

    def regenerate(self, nonconverging, new_params):
        """
        regenerate the nonconverging params
        """
        loadings = self.info["loadings"]
        for loading in loadings:
            for indexList, indexNonconvergingParam in enumerate(nonconverging):
                index = str(indexNonconvergingParam)
                new_param = new_params[indexList]
                simPath = f"{self.simulationPath}/{loading}/{index}"
                if os.path.exists(simPath) and os.path.isdir(simPath):
                    shutil.rmtree(simPath)
                self.make_new_initial_job(new_param, loading, index)

    def save_initial_outputs(self):
        loadings = self.info["loadings"]
        material = self.info["material"]
        exampleLoading = self.info["exampleLoading"]
        convertUnit = self.info["convertUnit"]
        np.save(f'{self.resultPath}/initial_params.npy', list(self.path2params[exampleLoading].values()))
        for loading in loadings:
            self.initial_trueCurves[loading] = {}
            self.initial_processCurves[loading] = {}
            for (index, params) in self.path2params[loading].items():
                simPath = f"{self.simulationPath}/{loading}/{index}"
                path2txt = f'{simPath}/postProc/{material}_tensionX.txt'
                if loading.startswith("linear"):
                    processCurves = preprocessDAMASKLinear(path2txt)
                    processCurves["stress"] *= 1e-6
                    trueCurves = preprocessDAMASKTrue(path2txt)
                    trueCurves["stress"] *= 1e-6
                else: 
                    processCurves = preprocessDAMASKNonlinear(path2txt)
                    processCurves["stress"] *= 1e-6
                    trueCurves = preprocessDAMASKTrue(path2txt)
                    trueCurves["stress"] *= 1e-6
                self.initial_processCurves[loading][tuple(params.items())] = processCurves
                self.initial_trueCurves[loading][tuple(params.items())] = trueCurves
            np.save(f"{self.resultPath}/{loading}/initial_processCurves.npy", self.initial_processCurves[loading])
            np.save(f"{self.resultPath}/{loading}/initial_trueCurves.npy", self.initial_trueCurves[loading]) 

        np.save(f"{self.resultPath}/initial_allLoadings_processCurves.npy", self.initial_processCurves)
        np.save(f"{self.resultPath}/initial_allLoadings_trueCurves.npy", self.initial_trueCurves) 

    #########################
    # ITERATION SIMULATIONS #
    #########################

    def run_iteration_simulations(self, dictParams):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        loadings = self.info['loadings']
        curveIndex = self.info["curveIndex"]
        optimizerName = self.info["optimizerName"]
        searchingSpace = self.info["searchingSpace"]

        self.simulationPath = f"simulations/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}"
        self.resultPath = f"results/{material}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}"

        for loading in loadings:
            self.path2params[loading] = {}
            simPath = f"{self.simulationPath}/{loading}"
            for fileIndex in os.listdir(simPath):
                if fileIndex == str(self.fileIndex):
                    shutil.rmtree(f"{simPath}/{fileIndex}")
            fileIndex = str(self.fileIndex)
            self.make_new_iteration_job(dictParams, loading, fileIndex)
        converging = self.submit_iteration_jobs()
        if converging:
            self.save_iteration_outputs()
            return (True, self.iteration_trueCurves, self.iteration_processCurves)
        else:
            return (False, None, None)

    def make_new_iteration_job(self, params, loading, fileIndex):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        simPath = f"{self.simulationPath}/{loading}/{fileIndex}"
        shutil.copytree(f"templates/{material}/{CPLaw}/{loading}", simPath) 
        self.path2params[loading][fileIndex] = params
        if self.info["CPLaw"] == "PH":
            self.edit_material_parameters_PH(params, simPath)
        if self.info["CPLaw"] == "DB":
            self.edit_material_parameters_DB(params, simPath)

    def submit_iteration_jobs(self):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        projectPath = self.info['projectPath']
        loadings = self.info["loadings"]
        server = self.info["server"]
        curveIndex = self.info["curveIndex"]
        exampleLoading = self.info["exampleLoading"]
        numberOfLoadings = len(loadings)
   
        with open(f"linux_slurm/array_{CPLaw}{curveIndex}_file.txt", 'w') as filename:
            for loading in loadings:
                filename.write(f"{projectPath}/{self.simulationPath}/{loading}/{self.fileIndex}\n")
        subprocess.run(f"sbatch --wait --array=1-{numberOfLoadings} linux_slurm/{server}_array_pre.sh {material} {CPLaw}{curveIndex}", shell=True)
 
        nonconvergings = []
        for loading in loadings:
            simPath = f"{self.simulationPath}/{loading}"
            nonconvergings.extend(self.nonconvergingSimsIteration(simPath, loading))
        nonconvergings = list(set(nonconvergings))
        # Saving nonconvergin params
        for index in nonconvergings:
            self.nonconvergingParams.add(tuple(self.path2params[exampleLoading][str(index)].items()))
        
        # This iteration does not converge
        if len(nonconvergings) != 0:
            return False
        else: 
            with open(f"linux_slurm/array_{CPLaw}{curveIndex}_file.txt", 'w') as filename:
                for loading in loadings:
                    filename.write(f"{projectPath}/{self.simulationPath}/{loading}/{self.fileIndex}\n")
            subprocess.run(f"sbatch --wait --array=1-{numberOfLoadings} linux_slurm/{server}_array_post.sh {material} {CPLaw}{curveIndex}", shell=True)
            return True

    def save_iteration_outputs(self):
        loadings = self.info["loadings"]
        material = self.info["material"]

        for loading in loadings:
            self.iteration_trueCurves[loading] = {}
            self.iteration_processCurves[loading] = {}

            index = list(self.path2params[loading].items())[0][0]
            params = list(self.path2params[loading].items())[0][1]
            simPath = f"{self.simulationPath}/{loading}/{index}"
            path2txt = f'{simPath}/postProc/{material}_tensionX.txt'
            if loading.startswith("linear"):
                processCurves = preprocessDAMASKLinear(path2txt)
                trueCurves = preprocessDAMASKTrue(path2txt)
            else: 
                processCurves = preprocessDAMASKNonlinear(path2txt)
                trueCurves = preprocessDAMASKTrue(path2txt)
            self.iteration_processCurves[loading][tuple(params.items())] = processCurves
            self.iteration_trueCurves[loading][tuple(params.items())] = trueCurves

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
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("dipmin", param_dict))
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
            if lines[i].startswith("D0"):
                lines[i] = self.replaceAllNumbersInLine(lines[i], self.checkReplace("D0", param_dict))    
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

    def latin_hypercube_sampling(self, numberOfSims):
        points = []
        np.random.seed(20)
        param_info = self.info["param_info"]
        #print(param_info)


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

