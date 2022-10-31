import math
import random
import time
import threading
from copy import deepcopy

from diversipy import *

from scipy.optimize import minimize, fmin_l_bfgs_b
from scipy.spatial import distance
import scipy.stats

import numpy as np
from sklearn.metrics import euclidean_distances

from GeneticAlgorithmOptimizer import GeneticAlgorithm
from ParticleSwarmOptimizer import ParticleSwarmOptimizer
from ThompsonMAB import MyBayesianUCBBandit, MyEXP3, MyThompsonSamplingBandit, MyGaussianTS, MyNormalDistributionThompsonSamplingBandit
from core.BO.gaussian_process import GaussianProcess, CostGaussianProcess
from core.BO.acquisition_functions import AcquisitionFunction

import multiprocessing

from core.DIRECT import DIRECTAlgo
from pyDOE2 import lhs

import pandas as pd

import matplotlib.pyplot as plt

import sklearn.gaussian_process as gpLib
import gen_func

num_cores = multiprocessing.cpu_count()

num_of_trainingPoints = 40

X = []
Xobs = []
Yobs = []

class CostBayesianOptimizer():
    def __init__(self, inFunc, initGuess=10, lamda=0.1, kerneltype="rbf", numIter=60, localOptAlgo="L-BFGS-B", useLib=True, MABmethod="BayesianUCB", trial=None):
        self.logs = ""
        self.logDup = ""
        self.logNext = ""
        self.csvLog = ""
        self.trial = trial
        self.logEps = ""
        self.nInitGuess = initGuess
        self.Xobs = []
        self.Yobs = []
        self.function = inFunc
        self.numMultiStart = 10 * (inFunc.input_dim * inFunc.input_dim)

        self.lbfgs_gtol = 1e-05
        self.lbfgs_ftol = 1e-03

        self.localOptimAlgo = localOptAlgo

        self.ite = 0
        self.maxIter = numIter

        self.nextArray = []
        self.dup = 0
        self.suggestDup = 0
        self.regretArray = []

        self.dropbest = 0
        self.kernelType = kerneltype
        self.useLib = useLib

        ### Manually-tuned hyperparameters ###
        # Branin
        if inFunc.name == "Branin":
            self.GP_default_hand_tuned_len = 0.27
            self.GP_default_hand_tuned_amp = 6.86
        # GriewankNd
        if inFunc.name == "GriewankNd": #2d
            self.GP_default_hand_tuned_len = 0.028
            self.GP_default_hand_tuned_amp = 0.446
        # Mccormick2d
        if inFunc.name == "Mccormick2d":
            self.GP_default_hand_tuned_len = 0.456 #0.47, 0.405, 0.447, 0.5
            self.GP_default_hand_tuned_amp = 3.5 #5.11, 1.021, 3.6, 4.315
        # Schubert2d
        if inFunc.name == "Schubert2d":
            self.GP_default_hand_tuned_len = 0.108
            self.GP_default_hand_tuned_amp = 4.46
        if inFunc.name == "Eggholder":
            self.GP_default_hand_tuned_len = 0.046
            self.GP_default_hand_tuned_amp = 0.91
        if inFunc.name == "Test1d":
            self.GP_default_hand_tuned_len = 0.055
            self.GP_default_hand_tuned_amp = 0.62
        if inFunc.name == "AckleyNd":
            self.GP_default_hand_tuned_len = 0.14
            self.GP_default_hand_tuned_amp = 0.66
        if inFunc.name == "AlpineNd":
            self.GP_default_hand_tuned_len = 0.07
            self.GP_default_hand_tuned_amp = 0.99
        if inFunc.name == "RastriginNd":
            self.GP_default_hand_tuned_len = 0.049
            self.GP_default_hand_tuned_amp = 0.67
        if inFunc.name == "Hartman_6d":
            self.GP_default_hand_tuned_len = 0.18
            self.GP_default_hand_tuned_amp = 0.99
        if inFunc.name == "Michalewicz10d":
            self.GP_default_hand_tuned_len = 0.2
            self.GP_default_hand_tuned_amp = 0.99
        if inFunc.name == "RealExperiment01":
            self.GP_default_hand_tuned_len = 0.01
            self.GP_default_hand_tuned_amp = 0.99
        if inFunc.name == "RealExperiment00":
            self.GP_default_hand_tuned_len = 0.01
            self.GP_default_hand_tuned_amp = 0.99
            hidden_layer1_var = [8, 16, 32, 64, 128, 256, 512]
            hidden_layer2_var = [8, 16, 32, 64, 128, 256, 512]
            learning_rate_init = [0.0001, 0.001, 0.01, 0.1]
            batch_size = [8, 16, 32, 64]

            self.candidatePoints = []
            for v1 in hidden_layer1_var:
                for v2 in hidden_layer2_var:
                    for v3 in learning_rate_init:
                        for v4 in batch_size:
                            self.candidatePoints.append([v1, v2, v3, v4])
        if inFunc.name == "RealExperiment02" or inFunc.name == "RealExperiment03":
            self.GP_default_hand_tuned_len = 0.01
            self.GP_default_hand_tuned_amp = 0.99
            xcors = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]  # 7
            ycors = [-1.1, -0.5, 0.0, 0.5, 1.1]  # 5
            depths = [4330.9624, 4400.0, 4500.0, 4600.0, 4700.0, 4800.0, 4900.0, 5000.0, 5100.0, 5200.0, 5300.0,
                      5467.4624]  # 12
            vels = [200, 300, 400, 500, 600, 700]  # 6
            self.candidatePoints = []
            for v1 in xcors:
                for v2 in ycors:
                    for v3 in depths:
                        for v4 in vels:
                            self.candidatePoints.append([v1, v2, v3, v4])

        # Define acquisition function
        acq_func = {}
        acq_func['dim'] = self.function.input_dim
        acq_func['name'] = "ei"  # Default acquisition function
        acq_func['epsilon'] = 0.01

        self.acquisition = AcquisitionFunction(acq_func)
        self.acquisition.setEsigma(lamda)

        # Choose kernel
        if self.useLib:
            self.GP = GaussianProcess('rbf', self.function.input_dim, self.function, isUseLib=True)  # Choose kernel
            self.costGP = GaussianProcess('rbf', self.function.input_dim, self.function, isUseLib=True)
        else:
            self.GP = GaussianProcess('rbf', self.function.input_dim, self.function, isUseLib=False)
            self.costGP = GaussianProcess('linear', self.function.input_dim, self.function, isUseLib=True)
            self.costGP.name = "CostGP"
            ### Fix hyperparams of kernelse
            self.GP.kernel.lengthScale = self.GP_default_hand_tuned_len
            self.GP.kernel.amp = self.GP_default_hand_tuned_amp
            self.costGP.kernel.lengthScale = self.GP_default_hand_tuned_len
            self.costGP.kernel.amp = self.GP_default_hand_tuned_amp

            self.GP.optimizeHyperparams = True
            self.myProposedGP = CostGaussianProcess('cost_rbf', self.function.input_dim, self.function, self.trial, isUseLib=False)
            self.myProposedGP.kernel.lengthScal = self.GP_default_hand_tuned_len
            self.myProposedGP.kernel.amp = self.GP_default_hand_tuned_amp
            self.costGP.setCostFunction(self.function)
            self.myProposedGP.setCostGP(self.costGP)

        self.scaleBeta = 1.0
        self.inLamda = lamda

        self.DIRECT = DIRECTAlgo()

        self.directIte = 200 * self.function.input_dim
        self.directEval = 200 * self.function.input_dim
        self.directDeep = 200 * self.function.input_dim

        ### PSO Optimizer ###
        self.num_particles = 40
        self.PSOptimizer = ParticleSwarmOptimizer()

        self.acquisition.setDim(self.function.input_dim)

        self.log_iter = ""
        self.logEveryTen = ""

        numOfAmrs = 2

        self.MABmethod = MABmethod
        if self.MABmethod == "BayesianUCB":
            # self.myTSB = MyBayesianUCBBandit(numOfAmrs)
            self.myTSB = MyNormalDistributionThompsonSamplingBandit(numOfAmrs)
        elif self.MABmethod == "TS":
            self.myTSB = MyThompsonSamplingBandit(numOfAmrs)
        elif self.MABmethod == "EXP3":
            self.myTSB = MyEXP3(numOfAmrs,0.1)
        elif self.MABmethod == "GaussianTS":
            self.myTSB = MyGaussianTS(numOfAmrs,self.function.nbounds, self.function.randUniformInNBounds, self.function)

        self.turns = []
        self.rewards = []

        self.f_xs = []
        self.c_xs = []

        self.gs = [0.0, 0.0]
        self.ps = [0.0, 0.0]
        self.hedges = [0, 0]

        self.eta = 1.0

    def setLamda(self,val):
        self.inLamda = val

    def setRun(self, val):
        self.trial = val
        self.myProposedGP.setTrial(val)

    def resetBO(self):
        self.Xobs = []
        self.Yobs = []
        self.nXobs = []
        self.nYobs = []
        self.Yobs_eval = []
        self.Yobs_cost = []

        self.ite = 0

        self.nextArray = []
        self.dup = 0
        self.suggestDup = 0
        self.regretArray = []

        self.dropbest = 0

        if self.useLib:
            self.GP = GaussianProcess('rbf', self.function.input_dim, self.function, isUseLib=True)  # Choose kernel
            self.costGP = GaussianProcess('rbf', self.function.input_dim, self.function, isUseLib=True)
        else:
            self.GP = GaussianProcess('rbf', self.function.input_dim, self.function, isUseLib=False)
            self.costGP = GaussianProcess('linear', self.function.input_dim, self.function, isUseLib=True)

            ### Fix hyperparams of kernels
            self.GP.kernel.lengthScale = self.GP_default_hand_tuned_len
            self.GP.kernel.amp = self.GP_default_hand_tuned_amp

            self.costGP.kernel.lengthScale = self.GP_default_hand_tuned_len
            self.costGP.kernel.amp = self.GP_default_hand_tuned_amp

            self.myProposedGP = CostGaussianProcess('cost_rbf', self.function.input_dim, self.function, self.trial,
                                                    isUseLib=False)
            self.myProposedGP.kernel.lengthScal = self.GP_default_hand_tuned_len
            self.myProposedGP.kernel.amp = self.GP_default_hand_tuned_amp
            self.costGP.setCostFunction(self.function)
            self.myProposedGP.setCostGP(self.costGP)
        self.function.numCalls = 0

        self.turns = []
        self.rewards = []

        self.gs = [0.0, 0.0]
        self.ps = [0.0, 0.0]

        self.log_iter = ""
        self.logEveryTen = ""

    def localOptimize(self, input):
        mini = minimize(lambda x: -self.acquisition.acq_kind(x, input[0], input[1], obs=input[4], costGP=input[5], normalGP=input[6]), input[2], bounds=input[3], method="L-BFGS-B", options={'ftol': self.lbfgs_ftol, 'gtol': self.lbfgs_gtol})
        tmpNextX = (mini.x[:]).tolist()
        tmpY = self.acquisition.acq_kind(np.array(tmpNextX), input[0], input[1], obs=input[4], costGP=input[5], normalGP=input[6])
        return tmpNextX, tmpY

    def _init_guess(self, numOfGuess):
        self.Xobs = []
        for i in range(numOfGuess):
            self.Xobs.append(self.function.randUniformInBounds())

        self.Yobs = [self.function.func(np.array(i)) for i in self.Xobs]
        self.nXobs = self.Xobs.copy()
        for i in range(0,len(self.nXobs)):
            self.nXobs[i] = self.function.normalize(self.nXobs[i])

    def cost_effective_initial_design(self, initBudget):
        if self.function.name == "RealExperiment00" or self.function.name == "RealExperiment02" or self.function.name == "RealExperiment03":
            ########################################
            initCandidate = [ins for ins in self.Xobs]
            testXobs = self.Xobs
            testnXobs = [self.function.normalize(ins) for ins in testXobs]
            testYobs_eval = self.Yobs_eval
            testYobs_cost = self.Yobs_cost
            ct = 0

            design = [[float(val) for val in ins] for ins in self.function.ncandidatePoints]
            candidatesPool = [deepcopy(ins) for ins in design]

            while ct < initBudget:
                # while self.ite < self.maxIter:
                self.costGP.fit(testnXobs, testYobs_cost)

                ### Assume cost func is unknown
                predicted_costs = [self.costGP.predict(ins)[0] for ins in candidatesPool]
                ### Known cost function
                cost_candidate = sorted(zip(predicted_costs, candidatesPool))

                if len(initCandidate) < 1:
                    remainingX = cost_candidate[0][1]
                else:
                    while len(cost_candidate) > 1:
                        cost_candidate.pop()  # Excluse the most expensive point
                        if len(cost_candidate) == 1:
                            break
                        # Find the closest point
                        min_dis = np.inf
                        min_idx = -1
                        # print("Test cost can:",cost_candidate)
                        for i in range(len(cost_candidate)):
                            distance_of_ins = np.inf
                            for initIns in testnXobs:
                                dis = np.linalg.norm(np.array(cost_candidate[i][1]) - np.array(initIns), 2)
                                if dis < distance_of_ins:
                                    distance_of_ins = dis
                            if distance_of_ins <= min_dis:
                                min_dis = distance_of_ins
                                min_idx = i
                        # print("min_idx:",min_idx, " len:", len(cost_candidate))
                        candidatesPool.remove(cost_candidate[min_idx][1])
                        cost_candidate.pop(min_idx)
                        if len(cost_candidate) == 1:
                            break

                # Evaluate the remaining point and update cost surrogate model
                remainingX = cost_candidate[0][1]
                initCandidate.append(remainingX)
                denorm = [ins for ins in self.function.denormalize(remainingX)]
                if self.function.name == "RealExperiment00":
                    denorm[0] = str(int(denorm[0]))
                    denorm[1] = str(int(denorm[1]))
                    denorm[2] = str(denorm[2])
                    denorm[3] = str(int(denorm[3]))
                elif self.function.name == "RealExperiment02" or self.function.name == "RealExperiment03":
                    denorm[0] = str(denorm[0])
                    denorm[1] = str(denorm[1])
                    denorm[2] = str(denorm[2])
                    denorm[3] = str(int(denorm[3]))
                remainingY, cost = self.function.func(denorm, self.trial)
                ct += cost
                testXobs.append(self.function.denormalize(remainingX))
                testnXobs.append(remainingX)

                testYobs_eval.append(remainingY)
                testYobs_cost.append(cost)

            return testXobs, testYobs_eval, testYobs_cost
            ########################################

        initCandidate = [ins for ins in self.Xobs]


        testXobs = self.Xobs
        testnXobs = [self.function.normalize(ins) for ins in testXobs]
        testYobs_eval = self.Yobs_eval
        testYobs_cost = self.Yobs_cost
        ct = 0

        while ct < initBudget:
            self.costGP.fit(testnXobs, testYobs_cost)

            numpoints = 1000
            design = transform_spread_out(lhd_matrix(numpoints, self.function.input_dim))  # create latin hypercube design

            ### Assume cost func is unknown
            candidatesPool = [(deepcopy(ins)).tolist() for ins in design]
            predicted_costs = [self.costGP.predict(ins)[0] for ins in candidatesPool]
            ### Known cost function
            cost_candidate = sorted(zip(predicted_costs, candidatesPool))

            if len(testnXobs) < 1:
                remainingX = cost_candidate[0][1]
            else:
                while len(cost_candidate) > 1:
                    cost_candidate.pop() # Excluse the most expensive point
                    if len(cost_candidate) == 1:
                        break
                    # Find the closest point
                    min_dis = np.inf
                    min_idx = -1
                    # print("Test cost can:",cost_candidate)
                    for i in range(len(cost_candidate)):
                        distance_of_ins = np.inf
                        for initIns in testnXobs:
                            dis = np.linalg.norm(np.array(cost_candidate[i][1]) - np.array(initIns),2)
                            if dis < distance_of_ins:
                                distance_of_ins = dis
                        if distance_of_ins <= min_dis:
                            min_dis = distance_of_ins
                            min_idx = i
                    # print("min_idx:",min_idx, " len:", len(cost_candidate))
                    cost_candidate.pop(min_idx)
                    if len(cost_candidate) == 1:
                        break

            # Evaluate the remaining point and update cost surrogate model
            remainingX = cost_candidate[0][1]
            initCandidate.append(remainingX)
            # remainingY, cost = self.function.func(self.function.denormalize(remainingX))
            remainingY, cost = self.function.func(self.function.denormalize(remainingX), self.trial)
            ct += cost
            testXobs.append(self.function.denormalize(remainingX))
            testnXobs.append(remainingX)
            # testnXobs.append(self.function.normalize(remainingX))
            testYobs_eval.append(remainingY)
            testYobs_cost.append(cost)

        # return initCandidate
        return testXobs, testYobs_eval, testYobs_cost

    def run(self, **args):
        self.acquisition.acq_name = args.get("method")
        self.acquisition.setRun(self.trial)
        self.inputBudget = args.get("budget")
        # print("ACQ:", self.acquisition.acq_name)
        if args.get("method") == "ei-cool" or args.get("method") == "ei-cool-d":
            inX = args.get("Xobs")
            inYeval = args.get("Yobs_eval")
            inYcost = args.get("Yobs_cost")
            self.randomInitBudget = np.sum(inYcost)
            # self.Xobs = candidates[0]
            # self.Yobs_eval = candidates[1]
            # self.Yobs_cost = candidates[2]

            self.Xobs = inX
            self.Yobs_eval = inYeval
            self.Yobs_cost = inYcost

            if args.get("method") == "ei-cool-d":
                candidates, fcandidates, costcandidates = self.cost_effective_initial_design(self.inputBudget/8)
                for idx in range(len(candidates)):
                    self.Xobs.append(candidates[idx])
                    self.Yobs_eval.append(fcandidates[idx])
                    self.Yobs_cost.append(costcandidates[idx])

            self.nXobs = [self.function.normalize(xi) for xi in self.Xobs]

        elif args.get("method") == "proposed" or args.get("method") == "proposed-d":
            inX = args.get("Xobs")
            inYeval = args.get("Yobs_eval")
            inYcost = args.get("Yobs_cost")
            self.randomInitBudget = np.sum(inYcost)

            self.Xobs = inX
            self.Yobs_eval = inYeval
            self.Yobs_cost = inYcost
            # Normalize X
            # self.nXobs = [self.function.normalize(xi) for xi in self.Xobs]
            if args.get("method") == "proposed-d":
                self.acquisition.acq_name = "proposed"
                candidates, fcandidates, costcandidates = self.cost_effective_initial_design(self.inputBudget/8)
                for idx in range(len(candidates)):
                    self.Xobs.append(candidates[idx])
                    self.Yobs_eval.append(fcandidates[idx])
                    self.Yobs_cost.append(costcandidates[idx])
            self.nXobs = [self.function.normalize(xi) for xi in self.Xobs]
        else:
            if args.get("randomlyInit"):
                self._init_guess(self.nInitGuess)
            else:
                inX = args.get("Xobs")
                inYeval = args.get("Yobs_eval")
                inYcost = args.get("Yobs_cost")
                self.randomInitBudget = np.sum(inYcost)

                self.Xobs = inX
                self.Yobs_eval = inYeval
                self.Yobs_cost = inYcost
                # Normalize X
                self.nXobs = [self.function.normalize(xi) for xi in self.Xobs]
        return self._runBO()

    def _runBO(self):
        self.useOwnGradientBasedOptimizer = False
        # self.acquisition.acq_name = 'ei-cool'

        self.ite = 0
        lb = []
        nlb = []
        ub = []
        nub = []
        for i in self.function.bounds:
            lb.append(i[0])
            ub.append(i[1])
        for i in self.function.nbounds:
            nlb.append(i[0])
            nub.append(i[1])
        self.logDebug = ""

        cons = []
        for factor in range(len(self.function.nbounds)):
            lower = self.function.nbounds[factor][0]
            upper = self.function.nbounds[factor][1]
            l = {'type': 'ineq',
                 'fun': lambda x, lb=lower, i=factor: x[i] - lb}
            u = {'type': 'ineq',
                 'fun': lambda x, ub=upper, i=factor: ub - x[i]}
            cons.append(l)
            cons.append(u)
        directBounds = np.array(self.function.nbounds)
        self.OptimumIte = -1
        current_OptimumEvaluation = np.inf
        # global current_OptimumEvaluation
        current_OptimumCost = np.inf

        deltaX = 0.0
        if self.function.input_dim > 1:
            for tmpDim in range(0, self.function.input_dim):
                deltaX += 0.5 * 0.5
            deltaX = np.sqrt(deltaX)
        else:
            deltaX = 0.5

        deltaX = deltaX / self.inLamda

        delt = 0.1
        a = 1.0
        b = 1.0
        dim = self.function.input_dim
        r = 1.0
        for ibound in self.function.bounds:
            r = max(r, ibound[1] - ibound[0])

        # For n-dim function
        bnds = ()  # Define boundary tuple
        for i in self.function.bounds:
            bnds = bnds + (i,)
        nbnds = ()  # Define boundary tuple
        for i in self.function.nbounds:
            nbnds = nbnds + (i,)

        # printit()
        totalBudget = self.inputBudget
        remainedBudget = totalBudget # in seconds

        initBudget = np.sum(self.Yobs_cost) - self.randomInitBudget
        usedBudget = initBudget
        remainedBudget -= usedBudget
        print("initBudget:", initBudget, " usedBuget:", usedBudget, " remainedBudget:", remainedBudget)
        # print("Obs:", len(self.Xobs))
        # exit()

        turn = 0
        trackBudget = 0
        self.initObsX = [deepcopy(init_obsX) for init_obsX in self.nXobs]
        self.initObsY = [deepcopy(init_obsY) for init_obsY in self.Yobs_eval]

        # Budget from init observations
        log_dif = np.log((self.function.ismax * self.function.fmin) - np.max(self.Yobs_eval) + 1e-8)
        while usedBudget >= trackBudget:
            # print(str(np.max(self.Yobs_eval)) + "," + str(trackBudget), ",", str(self.ite))
            self.log_iter += str(np.max(self.Yobs_eval)) + "," + str(trackBudget) + "," + str(
                log_dif) + "," + str(len(self.nXobs)) + '\n'
            self.logEveryTen += str(np.max(self.Yobs_eval)) + "," + str(trackBudget) + "," + str(
                log_dif) + "," + str(len(self.nXobs)) + '\n'
            trackBudget += 50

        while (remainedBudget > 0):
        # while self.ite < self.maxIter:

            #####################Theorem1 Srinivas#####################
            precalBetaT = 2.0 * np.log((self.ite + 1) * (self.ite + 1) * math.pi ** 2 / (3 * delt)) + 2 * dim * np.log(
                (self.ite + 1) * (self.ite + 1) * dim * b * r * np.sqrt(np.log(4 * dim * a / delt)))
            BetaT = np.sqrt(precalBetaT) / self.scaleBeta
            ###########################################################
            self.acquisition.setBetaT(BetaT)

            alpha = (totalBudget - usedBudget) / (totalBudget - initBudget)
            if self.acquisition.acq_name == 'ei' or self.acquisition.acq_name == 'ei_pure':
                turn = self.myTSB.suggestBandit()

                self.turns.append(turn)
                self.acquisition.setTurn(turn)

                self.GP.fit(self.nXobs, self.Yobs_eval)
            elif self.acquisition.acq_name == 'eips' or self.acquisition.acq_name == 'eips_pure' or self.acquisition.acq_name == "ei_ucbps"\
                    or self.acquisition.acq_name == "ei-cool_ei" or self.acquisition.acq_name == 'eips_pure_sep' \
                    or self.acquisition.acq_name == "ucbc":
                if self.acquisition.acq_name == 'eips_pure':
                    maxcost = np.max(self.Yobs_cost)
                    mincost = np.min(self.Yobs_cost)

                    turn = self.myTSB.suggestBandit()

                elif self.acquisition.acq_name == 'eips_pure_sep':
                    maxcost = np.max(self.Yobs_cost)
                    mincost = np.min(self.Yobs_cost)
                    gap = 1.0*(maxcost - mincost)/3.0
                    optimum_at_cost = self.Yobs_cost[np.argmax(self.Yobs_eval)]
                    if optimum_at_cost < (mincost+gap):
                        turn = 0
                    else:
                        turn = 1

                self.turns.append(turn)
                self.acquisition.setTurn(turn)
                self.acquisition.setAlpha(alpha)

                # Fit observed X and Y into evaluating GP
                self.GP.fit(self.nXobs, self.Yobs_eval)
                # Fit observed X and Y into cost GP
                self.costGP.fit(self.nXobs, self.Yobs_cost)
            elif self.acquisition.acq_name == 'ei-cool' or self.acquisition.acq_name == 'ei-cool-d':
                # Fit observed X and Y into evaluating GP
                print("self.nXobs size:", len(self.nXobs))
                self.GP.fit(self.nXobs, self.Yobs_eval)
                # Fit observed X and Y into cost GP
                self.costGP.fit(self.nXobs, self.Yobs_cost)

                self.acquisition.setAlpha(alpha)
                # self.myProposedGP.libKern.setAlpha(alpha)
                # print("Alpha:", alpha)
            elif self.acquisition.acq_name == "proposed" or self.acquisition.acq_name == "proposed-d" or self.acquisition.acq_name == "proposed_pure":
                turn = self.myTSB.suggestBandit()
                #### If use Round Robin 1/1 ####
                # turn = self.ite
                #### If use Round Robin 2/3 or 1/3 ####
                # turn += 1
                # if turn == 3:
                #     turn = 0
                self.turns.append(turn)
                print("Turn:", turn)

                alpha = (totalBudget - usedBudget) / (totalBudget - initBudget)
                self.GP.fit(self.nXobs, self.Yobs_eval)

                self.costGP.fit(self.nXobs, self.Yobs_cost)

                self.myProposedGP.kernel.setAlpha(alpha)
                self.myProposedGP.kernel.setTurn(turn)
                self.myProposedGP.libKern.setCostGP(self.costGP)
                self.myProposedGP.kernel.setCostGP(self.costGP)
                self.myProposedGP.setCostGP(self.costGP)
                self.myProposedGP.fit(self.nXobs, self.Yobs_eval)

                self.acquisition.setTurn(turn)

            elif self.acquisition.acq_name == "GaussianTS":
                turn = self.myTSB.suggestBandit()
                print("turn:", turn)
                self.turns.append(turn)
                self.acquisition.setTurn(turn)

                self.GP.fit(self.nXobs, self.Yobs_eval)
                # Fit observed X and Y into cost GP
                self.costGP.fit(self.nXobs, self.Yobs_cost)
            elif self.acquisition.acq_name == "GP-Hedge":
                # Fit observed X and Y into evaluating GP
                self.GP.fit(self.nXobs, self.Yobs_eval)
                # Fit observed X and Y into cost GP
                self.costGP.fit(self.nXobs, self.Yobs_cost)

            # Find the current best point
            current_max_y = np.max(self.Yobs_eval)
            # current_max_cost = np.max([ins[1] for ins in self.Yobs])
            if current_max_y != current_OptimumEvaluation:
                current_OptimumEvaluation = current_max_y
                self.OptimumIte = self.ite
            # x0 = self.Xobs[np.argmax(self.Yobs_eval)]  # Current best optimal point
            x0 = self.nXobs[np.argmax(self.Yobs_eval)]  # Current best optimal point

            def suggest(BetaT, initX, inputGP):
                self.acquisition.setBetaT(BetaT)
                self.acquisition.setIte(self.ite + 1)
                ############### L-BFGS-B
                if self.localOptimAlgo == "L-BFGS-B":
                    mini = minimize(lambda x: -self.acquisition.acq_kind(x, inputGP, current_max_y, obs=self.nXobs, costGP=self.costGP, normalGP=self.GP),
                                    initX, bounds=nbnds, method="L-BFGS-B", options={'ftol': self.lbfgs_ftol, 'gtol': self.lbfgs_gtol})

                    suggestedX = (mini.x[:]).tolist()
                    # tmpAcq = self.acquisition.acq_kind(suggestedX, self.GP, current_max_y, obs=self.Xobs)
                    tmpAcq = mini.fun * -1.0
                    # ##########################
                    # randomPoints = []
                    # while (len(randomPoints) < self.numMultiStart):
                    #     initX = self.function.randUniformInNBounds()
                    #     while initX in randomPoints:
                    #         initX = self.function.randUniformInNBounds()
                    #     randomPoints.append(initX)
                    randomPoints = lhs(self.function.input_dim, samples=5,
                                       criterion="maximin").tolist()
                    tmp = []
                    for i in randomPoints:
                        optX, optA = self.localOptimize([inputGP, current_max_y, i, nbnds, self.nXobs, self.costGP, self.GP])
                        tmp.append([optX, optA])


                    tmpX = [tmpT[0] for tmpT in tmp]
                    tmpA = [tmpT[1] for tmpT in tmp]
                    tmpX.append(suggestedX)
                    tmpA.append(tmpAcq)

                    print("maxAcq:", np.max(tmpA))

                    suggestedX = tmpX[np.argmax(tmpA)].copy()
                    output_xobs = []

                if self.localOptimAlgo == "DIRECT":
                    # ########################## DIRECT ##########################
                    nextX, _, output_xobs, fc = self.DIRECT.minimize(
                        lambda x: -self.acquisition.acq_kind(x, inputGP, current_max_y, obs=self.nXobs, costGP=self.costGP, normalGP=self.GP), directBounds,
                        max_iters=self.directIte,
                        max_evals=self.directEval,
                        max_deep=self.directDeep)
                    suggestedX = (nextX[:]).tolist()
                    tempX = deepcopy(suggestedX)

                    opt_result = fmin_l_bfgs_b(func=lambda x: -self.acquisition.acq_kind(x, inputGP, current_max_y, obs=self.nXobs, costGP=self.costGP, normalGP=self.GP), x0=np.array(tempX),
                                               bounds=nbnds, pgtol=1e-37,approx_grad=True)
                    suggestedX = opt_result[0]

                if self.localOptimAlgo == "TS":
                    tmpAcqs = [np.asscalar(self.acquisition.acq_kind(self.function.normalize(ins), inputGP, current_max_y, obs=self.nXobs,
                                                         costGP=self.costGP, normalGP=self.GP)) if ins not in self.Xobs else -9999 for ins in self.candidatePoints]

                    suggestedX = deepcopy(self.candidatePoints[np.argmax(tmpAcqs)])

                    suggestedX = self.function.normalize(suggestedX)
                    if self.function.name == "RealExperiment00" or self.function.name == "RealExperiment02" or self.function.name == "RealExperiment03":
                        self.suggestedIdx = np.argmax(tmpAcqs)
                    output_xobs = []

                if self.localOptimAlgo == "PSO":
                    inputGP.ThisCall = 0
                    swarm_candidates = lhs(self.function.input_dim, samples=self.num_particles, criterion="maximin").tolist()
                    suggestedX, output_xobs = self.PSOptimizer.minimizeMatrix(lambda x: -self.acquisition.acq_kind(x, inputGP, current_max_y, obs=self.nXobs, costGP=self.costGP, normalGP=self.GP),
                                                    nlb, nub, f_ieqcons=None, maxiter=100, swarmsize=self.num_particles,
                                                    input_swarms=swarm_candidates, minstep=1e-8, minfunc=1e-8)

                if self.localOptimAlgo == "GA":
                    def wrapperObj(x):
                        res = np.array([self.acquisition.acq_kind(ins, inputGP, current_max_y, obs=self.nXobs, costGP=self.costGP, normalGP=self.GP) * -1.0 for ins in x]).ravel()
                        return res

                    self.GAOptimizer = GeneticAlgorithm(
                        function=wrapperObj,
                        dim=self.function.input_dim,
                        lb=nlb,
                        ub=nub,
                        int_var=[],
                        pop_size=max([2 * self.function.input_dim, 40]),
                        num_gen=100,
                    )
                    suggestedX, f_min = self.GAOptimizer.optimize()
                    output_xobs = []

                return suggestedX, output_xobs

            # Suggest a next point
            if self.acquisition.acq_name == "proposed":
                nextNX, mini_xobs = suggest(BetaT, x0, self.myProposedGP)
            elif self.acquisition.acq_name == "GP-Hedge":
                self.acquisition.acq_name = "ei"
                ei_nextNX, ei_mini_xobs = suggest(BetaT, x0, self.GP)
                self.acquisition.acq_name = "eips"
                eips_nextNX, eips_mini_xobs = suggest(BetaT, x0, self.GP)
                if self.hedges[0] < 3:
                    turn = 0
                    nextNX = ei_nextNX
                    reward_t, _, _ = self.GP.predict(nextNX)
                    self.gs[0] += reward_t
                    self.hedges[0]+=1
                elif self.hedges[1] < 3:
                    turn = 1
                    nextNX = eips_nextNX
                    reward_t, _, _ = self.GP.predict(nextNX)
                    self.gs[1] += reward_t
                    self.hedges[1] += 1
                else:
                    self.ps[0] = np.exp(self.eta * self.gs[0]) / (self.eta * np.exp(self.gs[0]) + self.eta * np.exp(self.gs[1]))
                    self.ps[1] = np.exp(self.eta * self.gs[1]) / (self.eta * np.exp(self.gs[0]) + self.eta * np.exp(self.gs[1]))
                    if self.ps[0] > self.ps[1]:
                        nextNX = ei_nextNX
                        turn = 0
                    else:
                        nextNX = eips_nextNX
                        turn = 1
                self.turns.append(turn)
                self.acquisition.acq_name = "GP-Hedge"
            else:
                nextNX, mini_xobs = suggest(BetaT, x0, self.GP)

            print("nextNX:", nextNX, " turn:", turn)
            # Query the true objective function
            if self.acquisition.acq_name == "proposed" or self.acquisition.acq_name == "proposed-d":
                nextX = self.function.denormalize(nextNX)
                if self.function.name == "RealExperiment00" or self.function.name == "RealExperiment02" or self.function.name == "RealExperiment03":
                    nextY, nextCost = self.function.func(self.function.candidatePoints[self.suggestedIdx], self.trial)
                else:
                    nextY, nextCost = self.function.func(np.array(nextX), self.trial)
            else:
                nextX = self.function.denormalize(nextNX)
                if self.function.name == "RealExperiment00" or self.function.name == "RealExperiment02" or self.function.name == "RealExperiment03":
                    nextY, nextCost = self.function.func(self.function.candidatePoints[self.suggestedIdx], self.trial)
                else:
                    nextY, nextCost = self.function.func(np.array(nextX), self.trial)

            if self.acquisition.acq_name == "proposed" or self.acquisition.acq_name == "proposed-d" or self.acquisition.acq_name == "proposed_pure"\
                    or self.acquisition.acq_name == "eips_pure" or self.acquisition.acq_name == "ei_ucbps" or self.acquisition.acq_name == "ei-cool_ei" \
                    or self.acquisition.acq_name == 'eips_pure_sep'  or self.acquisition == "ucbc":

                reward = 0
                evals = [ins for ins in self.Yobs_eval]
                evals.append(nextY)
                costs = [ins for ins in self.Yobs_cost]
                costs.append(nextCost)
                min_fxs = np.min(evals)
                max_fxs = np.max(evals)
                min_cxs = np.min(costs)
                max_cxs = np.max(costs)
                print("min fx:", min_fxs, " max fx:", max_fxs, " min cx:", min_cxs, " max cx:", max_cxs)
                fx12 = ((nextY - min_fxs)/(max_fxs - min_fxs)) * (2.0 - 1.0) + 1.0
                cx12 = ((nextCost - min_cxs) / (max_cxs - min_cxs)) * (2.0 - 1.0) + 1.0
                reward = ((fx12 / cx12) - 0.5) * (3.0 / 4.0)

                if self.MABmethod != "TS":
                    for iarm in range(self.myTSB.num_of_arms):
                        for idxR in range(len(self.myTSB.winsArray[iarm])):
                            fxir = self.myTSB.fx_s[iarm][idxR]
                            cxir = self.myTSB.cx_s[iarm][idxR]
                            fxir = ((fxir - min_fxs) / (max_fxs - min_fxs)) * (2.0 - 1.0) + 1.0
                            cxir = ((cxir - min_cxs) / (max_cxs - min_cxs)) * (2.0 - 1.0) + 1.0
                            rewardir = ((fxir / cxir) - 0.5) * (3.0 / 4.0)
                            self.myTSB.winsArray[iarm][idxR] = rewardir

                print("REWARD=", reward)
                self.rewards.append(reward)

                ## Simply r = eval - cost
                print("turn:", turn," rewards:",reward, " nextY:", nextY, " nextCost:",nextCost)
                if self.MABmethod == "TS":
                    reward = np.clip(reward,0.0,1.0)
                    r = 0.0
                    if nextY > np.max(self.Yobs_eval):
                        r = 1.0
                    self.myTSB.updateTrialsWins(turn, r)
                elif self.MABmethod == "BayesianUCB":
                    normalizedY = ((nextY - min_fxs)/(max_fxs - min_fxs))
                    r = normalizedY * 1.0*remainedBudget/totalBudget
                    # r = normalizedY
                    self.myTSB.updateTrialsWins(turn, r, nextY, nextCost)
                else:
                    self.myTSB.updateTrialsWins(turn, reward, nextY, nextCost)
            if self.acquisition.acq_name == "GaussianTS":
                reward = 0

                evals = [ins for ins in self.Yobs_eval]
                evals.append(nextY)
                costs = [ins for ins in self.Yobs_cost]
                costs.append(nextCost)

                reward = nextY
                print("REWARD:", reward)
                self.rewards.append(reward)
                self.myTSB.updateTrialsWins(turn, reward, nextNX, nextY, nextCost)
                print("arm 1:",self.myTSB.rewards[0])
                print("arm 2:", self.myTSB.rewards[1])

            # Calculate the remaining budget
            remainedBudget -= nextCost
            usedBudget += nextCost

            log_dif = np.log((self.function.ismax * self.function.fmin) - np.max(self.Yobs_eval) + 1e-8)
            while usedBudget > trackBudget:
                if usedBudget >= totalBudget:
                    break
                self.log_iter += str(np.max(self.Yobs_eval)) + "," + str(trackBudget) + "," + str(
                    log_dif) + "," + str(len(self.nXobs)) + '\n'
                self.logEveryTen += str(np.max(self.Yobs_eval)) + "," + str(trackBudget) + "," + str(
                    log_dif) + "," + str(len(self.nXobs)) + '\n'
                trackBudget += 50
                if trackBudget >= totalBudget:
                    break

            # Update Xobs Yobs
            try:
                nextNX = nextNX.tolist()
            except:
                pass
            self.nXobs.append(nextNX)
            self.Xobs.append(nextX)
            self.Yobs_eval.append(nextY)
            self.Yobs_cost.append(nextCost)

            if self.acquisition.acq_name == "GP-Hedge":
                self.GP.fit(self.nXobs, self.Yobs_eval)
                reward_t, _, _ = self.GP.predict(nextNX)
                self.gs[turn] += reward_t
                self.rewards.append(reward_t)
                self.turns.append(turn)


            # Calculate regret for future usages
            if np.nan in nextX:
                print("nextX:", nextX)
            print(self.ite, " Minimum: ", np.max(self.Yobs_eval) * self.function.ismax * -1.0, " used budget: \033[1m", usedBudget, "\033[0;0m at x:",
                  self.Xobs[np.argmax(self.Yobs_eval)], " nextX:", nextX, " nextY:", nextY * self.function.ismax,
                  " cost:", nextCost,
                  " Xobs size:", len(self.Xobs))

            # Increase counter
            self.ite += 1

        while(trackBudget < totalBudget):
            log_dif = np.log(self.function.ismax*self.function.fmin - np.max(self.Yobs_eval) + 1e-8)
            self.log_iter += str(np.max(self.Yobs_eval)) + "," + str(trackBudget)+ ","+ str(log_dif) + ","+ str(len(self.nXobs)) + '\n'
            self.logEveryTen += str(np.max(self.Yobs_eval)) + "," + str(trackBudget) + "," + str(
                log_dif) + "," + str(len(self.nXobs)) + '\n'
            trackBudget += 50
            if trackBudget >= totalBudget:
                break

        return self.Xobs[np.argmax(self.Yobs_eval)], self.Xobs, self.Yobs, self.ite, np.max(self.Yobs_eval)
