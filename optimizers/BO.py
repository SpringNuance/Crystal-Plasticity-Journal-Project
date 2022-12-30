from modules.stoploss import *
from modules.helper import *
from modules.preprocessing import *
import bayes_opt
from math import *
from optimizers.optimizer import *
import time 
class BO(optimizer):
    
    ##################################
    # OPTIMIZER CLASS INITIALIZATION #
    ##################################

    def __init__(self, info):
        self.info = info
        self.default_params = None
        self.optimize_params = None
        self.optimize_type = None
        
        #############################
        # Optimizer hyperparameters #
        #############################
        self.verbose = 0
        self.random_state = 1
        self.init_points = 100
        self.iterations = 200
        # Low kappa = 1 means more exploitation for UCB
        # High kappa = 10 means more exploration for UCB
        # Low xi = 0 means more exploitation for EI and POI
        # High xi = 0.1 means more exploration for EI and POI
        self.acquisition = "ucb"
        self.kappa = 3
        self.xi = 0.1
        self.alpha = 1

    ##########################
    # OPTIMIZATION FUNCTIONS #
    ##########################

    def multiply(self, tupleRange, multiplier):
        return tuple(int(i * multiplier) for i in tupleRange)

    def initializeOptimizer(self, default_params, optimize_params, optimize_type):
        loadings = self.info["loadings"]
        param_info = self.info["param_info"]
        param_info_BO = self.info["param_info_BO"]
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        exp_curves = self.info["exp_curves"] 
        regressors = self.info["regressors"]
        scalers = self.info["scalers"]
        weightsYielding = self.info["weightsYielding"]
        weightsHardening = self.info["weightsHardening"]
        weightsLoading = self.info["weightsLoading"]

        self.default_params = default_params
        # In BO, the parameter names need to be in alphabetical order
        self.optimize_params = sorted(optimize_params) 
        self.optimize_type = optimize_type 

        BO_bounds = {}
        for param in self.optimize_params:
            BO_bounds[param] = self.multiply(param_info_BO[param], 10 ** param_info[param]["round"])
        #print(pbounds)
        
        def fitnessBO(**solution):
            default_params_dict = dict(self.default_params)

            for param in self.optimize_params:
                scaledDownSolution = solution[param] * (10 ** - param_info[param]["round"])
                if searchingSpace == "discrete":
                    default_params_dict[param] = round_to_step(param_info[param]['low'], param_info[param]['step'], scaledDownSolution, param_info[param]['round'])
                elif searchingSpace == "continuous":
                    default_params_dict[param] = round(scaledDownSolution, roundContinuousDecimals)

            candidate_params = np.array(list(default_params_dict.values())) 

            if self.optimize_type == "yielding":
                scaledParams = scalers["linear_uniaxial_RD"].transform(candidate_params.reshape(1, -1))
                predicted_sim_stress = regressors["linear_uniaxial_RD"].predict(scaledParams).flatten() # changing [[1,2,3...]] into [1,2,3,..]
                fitness = fitnessYieldingLinear(exp_curves["interpolate"]["linear_uniaxial_RD"]["stress"], predicted_sim_stress, exp_curves["interpolate"]["linear_uniaxial_RD"]["strain"], weightsYielding)
            elif self.optimize_type == "hardening":
                fitness = 0
                for loading in loadings:
                    scaledParams = scalers[loading].transform(candidate_params.reshape(1, -1))
                    predicted_sim_stress = regressors[loading].predict(scaledParams).flatten() # changing [[1,2,3...]] into [1,2,3,..]
                    if loading == "linear_uniaxial_RD":
                        fitness += weightsLoading[loading] * fitnessHardeningLinear(exp_curves["interpolate"][loading]["stress"], predicted_sim_stress,  exp_curves["interpolate"][loading]["strain"], weightsHardening)
                    else:
                        fitness += weightsLoading[loading] * fitnessHardeningNonlinear(exp_curves["interpolate"][loading]["stress"], predicted_sim_stress,  exp_curves["interpolate"][loading]["strain"], weightsHardening)

            fitnessScore = 1/fitness
            return fitnessScore

        bo_instance = bayes_opt.BayesianOptimization(
            f = fitnessBO,
            pbounds = BO_bounds, 
            verbose = self.verbose,
            random_state = self.random_state
        )   
        self.optimizer = bo_instance

    def run(self):
        self.optimizer.maximize(
            init_points = self.init_points, 
            n_iter = self.iterations,    
            # What follows are GP regressor parameters
            acq=self.acquisition, 
            kappa=self.kappa, 
            xi=self.xi, 
            alpha=self.alpha)
 
        self.optimizer.set_gp_params(normalize_y=True)
        
    def outputResult(self):
        param_info = self.info["param_info"] 
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]

        solution = self.optimizer.max["params"]
        solution_fitness = self.optimizer.max["target"]
        
        solution_fitness = 1/solution_fitness
        
        solution_dict = dict(self.default_params)

        for param in self.optimize_params:
            scaledDownSolution = solution[param] * (10 ** - param_info[param]["round"])
            if searchingSpace == "discrete":
                solution_dict[param] = round_to_step(param_info[param]['low'], param_info[param]['step'], scaledDownSolution, param_info[param]['round'])
            elif searchingSpace == "continuous":
                solution_dict[param] = round(scaledDownSolution, roundContinuousDecimals)
        solution_tuple = tuple(solution_dict.items())
         
        output = {"solution_dict": solution_dict, "solution_tuple": solution_tuple, "solution_fitness": solution_fitness}
        return output       
