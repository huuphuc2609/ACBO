import math
import numpy as np
import scipy
from scipy.stats import norm

class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq):
        self.acq = acq
        self.acq_name = self.acq['name']
        self.betaT = 2.0

        self.epsilon = self.acq['epsilon']
        self.alpha = 1.0

        self.dens_good = None
        self.dens_bad = None
        self.varType = ""

    def setEpsilon(self, val):
        self.epsilon = val

    def setBetaT(self, val):
        self.betaT = val

    def setIte(self, val):
        self.ite = val

    def setDim(self, val):
        self.dim = val

    def setEsigma(self, val):
        self.esigma = val

    def setAlpha(self, val):
        self.alpha = val

    def setTurn(self, val):
        self.turn = val

    def setRun(self,val):
        self.trial = val

    @staticmethod
    def resampleFromKDE(kde, size):
        n, d = kde.data.shape
        indices = np.random.randint(0, n, size)
        cov = np.diag(kde.bw) ** 2
        means = kde.data[indices, :]
        norm = np.random.multivariate_normal(np.zeros(d), cov, size)
        return np.transpose(means + norm)

    def getEpsilon(self):
        return self.acq['epsilon']

    def acq_kind(self, x, gp, y_max, **args):
        # print("epsilon:", self.epsilon)
        # print("alpha:", self.alpha)
        if np.any(np.isnan(x)):
            return 0
        if self.acq_name == 'ei':
            return self._ei(x, gp, y_max, self.epsilon)
        if self.acq_name == 'ei_pure':
            if self.turn%2 == 0:
            # if self.turn == 0:
            # if self.turn == 0 or self.turn == 1:
                return self._ei(x, gp, y_max, self.epsilon)
            else:
                return self._pure_explore(x, gp)
        if self.acq_name == 'ei_ucbps':
            if self.turn == 0:
                return self._ei(x, gp, y_max, self.epsilon)
            elif self.turn == 1:
                return self._ucbps(x, gp, self.betaT, args.get("costGP"), self.trial)
        if self.acq_name == 'ucbc':
            # print("GOOOOOOOOOD")
            return self._ucbc(x, gp, self.betaT, args.get("costGP"), self.trial)
        if self.acq_name == 'eips':
            return self._eips(x, gp, y_max, self.epsilon, args.get("costGP"), self.trial)
        if self.acq_name == 'eips_set':
            return self._eips_set(x, gp, y_max, self.epsilon, args.get("costGP"), self.trial, self.alpha)
        if self.acq_name == 'eips_pure' or self.acq_name == "eips_pure_sep":
            # return self._eibs(x, gp, y_max, self.epsilon, args.get("costGP"), self.trial)
            # return self._explore_with_mean(x, gp, args.get("costGP"), self.trial)
            if self.turn == 0:
                # print("eips")
            # if self.turn == 0:
            # if self.turn == 0 or self.turn == 1:
                return self._eips(x, gp, y_max, self.epsilon, args.get("costGP"), self.trial)
            #     return self._pure_explore(x, gp)
            #     return self._eicool(x, gp, y_max, self.epsilon, args.get("costGP"), self.alpha, self.trial)
            elif self.turn == 1:
                # return self._eips(x, gp, y_max, self.epsilon, args.get("costGP"), self.trial)
                # print("pure explore")
                # return self._pure_explore(x, gp)
                # return self._explore_with_mean(x, gp, args.get("costGP"), self.trial)
                return self._ei(x, gp, y_max, self.epsilon)
                # return self._eibs(x, gp, y_max, self.epsilon, args.get("costGP"), self.trial)
            # elif self.turn == 2:
            #     return self._pure_explore(x, gp)
            #     return self._ei(x, gp, y_max, self.epsilon)
                # return self._eibscool(x, gp, y_max, self.epsilon, args.get("costGP"), self.alpha, self.trial)
        if self.acq_name == 'GaussianTS':
            if self.turn == 0:
                return self._eips(x, gp, y_max, self.epsilon, args.get("costGP"), self.trial)
                # return self._eicool(x, gp, y_max, self.epsilon, args.get("costGP"), self.alpha, self.trial)
                # return self._ei(x, gp, y_max, self.epsilon)
            elif self.turn == 1:
                # return self._pure_explore(x, gp)
                # return self._explore_with_mean(x, gp)
                return self._ei(x, gp, y_max, self.epsilon)
        if self.acq_name == 'ei-cool' or self.acq_name == 'ei-cool-d':
            return self._eicool(x, gp, y_max, self.epsilon, args.get("costGP"), self.alpha, self.trial)
        if self.acq_name == 'ei-cool_ei':
            if self.turn == 0:
                # print("eicool")
                return self._eicool(x, gp, y_max, self.epsilon, args.get("costGP"), self.alpha, self.trial)
            elif self.turn == 1:
                # print("ei")
                return self._ei(x, gp, y_max, self.epsilon)
        if self.acq_name == 'proposed':
            # return self._pure_explore(x, gp)
            return self._ei_proposed(x, gp, y_max, self.epsilon, args.get("costGP"))
        if self.acq_name == 'proposed_pure':
            # print("Chosen turn:", self.turn)
            if self.turn == 0: #### If use round robin 1/1
            # if self.turn == 0: #### If use round robin 1/3
            # if self.turn == 0 or self.turn == 1: #### If use round robin 2/3
                # print("Proposed!")
                return self._ei_proposed(x, gp, y_max, self.epsilon, args.get("costGP"))
            elif self.turn == 1:
                # print("Pure exploration!")
                return self._pure_explore(x, args.get("normalGP"))

                # return self._proposed_optional(x, args.get("normalGP"), args.get("costGP"), y_max, self.epsilon)
                # return self._pure_explore(x, gp)
            #     return self._pure_exploitation(x, gp) #BAD
            #     return self._ei(x, args.get("normalGP"), y_max, self.epsilon)
            #     return self._ei(x, gp, y_max, self.epsilon)
            #     return self._eips(x, args.get("normalGP"), y_max, self.epsilon, args.get("costGP"))
                # return self._eips(x, gp, y_max, self.epsilon, args.get("costGP")) #BAD


        if self.acq_name == 'pure_explore':
            return self._pure_explore(x, gp, y_max, self.epsilon, args.get("costGP"))

        if self.acq_name == 'ucb':
            return self._ucb(x, gp, self.acq['epsilon'])
        if self.acq_name == 'lcb':
            return self._lcb(x, gp, self.acq['kappa'])
        if self.acq_name == 'pi':
            return self._pi(x, gp, y_max, self.acq['epsilon'])

    @staticmethod
    def _pi(x, gp, fMax, epsilon):
        mean, _, var = gp.predict(x)
        #var[var < 1e-10] = 0
        std = np.sqrt(var)
        Z = (mean - fMax - epsilon) / std
        result = np.matrix(scipy.stats.norm.cdf(Z))
        return result

    @staticmethod
    def _ei(x, gp, fMax, epsilon):
        # print("ei")
        mean, _, var = gp.predict(x)
        # mean, _, var = gp.predictScalarLib(x)
        #mean, _, var = gp.predictScalarTrans(x)
        #print("mean: ", mean)
        #print("var: ", var)
        #mean, _, var = gp.predictCholeskyScalar(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        #var[var < 1e-10] = 0
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        # result = np.matrix((mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + np.multiply(np.matrix(std.ravel()), (
        #     np.matrix(scipy.stats.norm.pdf(Z)))).ravel())
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        return result

    @staticmethod
    def _ei_proposed(x, gp, fMax, epsilon, gp_cost):
        mean, _, var = gp.predict(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)

        #Cost
        # mean_cost, _, var_cost = gp_cost.predict(x)
        # if mean_cost <= 0:
        #     mean_cost = 1e-8
        # var2_cost = np.maximum(var_cost, 1e-4 + 0 * var_cost)
        # std_cost = np.sqrt(var2_cost)
        # Z_cost = (mean_cost - epsilon) / (std_cost)
        # result_cost = (mean_cost - epsilon) * scipy.stats.norm.cdf(Z_cost) + std_cost * scipy.stats.norm.pdf(Z_cost)
        # result_cost = mean_cost
        result_cost = 1

        result = result / result_cost
        return result

    @staticmethod
    def _pure_explore(x, gp):
        mean, _, var = gp.predict(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        std = np.sqrt(var2)

        # result = 0.5*mean + std (Good)
        result = std
        return result

    @staticmethod
    def _explore_with_mean(x, gp, gp_cost, trial):
        mean, _, var = gp.predict(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        std = np.sqrt(var2)
        # mean_cost = gp_cost.costFunction.func(gp_cost.costFunction.denormalize(x), trial)[1]
        result = mean + std
        return result

    @staticmethod
    def _pure_exploitation(x, gp):
        mean, _, var = gp.predict(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        std = np.sqrt(var2)
        result = mean
        return result

    @staticmethod
    def _proposed_optional(x, gp, gpcost, fMax, epsilon):
        mean, _, var = gp.predict(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)

        mean_cost, _, var_cost = gpcost.predict(x)
        var_cost2 = np.maximum(var_cost, 1e-4 + 0 * var_cost)
        std_cost = np.sqrt(var_cost2)
        result_cost = (mean_cost + std_cost)

        #result = mean*(1.0 - mean_cost/np.max(gpcost.Ym))
        result = result/result_cost
        return result

    @staticmethod
    def _eips(x, gp, fMax, epsilon, gp_cost, trial):
        mean, _, var = gp.predict(x)
        mean_cost, _, _ = gp_cost.predict(x)
        # mean_cost = gp_cost.costFunction.func(gp_cost.costFunction.denormalize(x), trial)[1]
        # print("eips:", mean_cost, " at x:", x)
        # def predictive_cost(x):
        #     res = 0
        #     # print("x:", x)
        #     # print("shape:", np.shape(x))
        #     insideX = gp_cost.func.denormalize(x)
        #     try:
        #         res += 10 + 5*insideX[0,1]
        #         # res += 50 + 15 * np.abs(x[0,0])
        #     except:
        #         res += 10 + 5 * insideX[1]
        #     return res
        #res
        # mean_cost = predictive_cost(x)

        # print("cost:", mean_cost)
        # mean_cost += 1e-8
        if mean_cost <= 0.0:
            mean_cost = 1e-8
        # mean, _, var = gp.predictScalarLib(x)
        # mean_cost, _, _ = gp_cost.predictScalarLib(x)

        # mean, _, var = gp.predictCholeskyScalar(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        # var[var < 1e-10] = 0
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        result = result/(mean_cost)
        return result

    @staticmethod
    def _eips_set(x, gp, fMax, epsilon, gp_cost, trial, alpha):
        mean, _, var = gp.predict(x)
        # mean_cost, _, _ = gp_cost.predict(x)
        mean_cost = gp_cost.costFunction.func(gp_cost.costFunction.denormalize(x), trial)[1]
        # print("eips:", mean_cost, " at x:", x)
        # def predictive_cost(x):
        #     res = 0
        #     # print("x:", x)
        #     # print("shape:", np.shape(x))
        #     insideX = gp_cost.func.denormalize(x)
        #     try:
        #         res += 10 + 5*insideX[0,1]
        #         # res += 50 + 15 * np.abs(x[0,0])
        #     except:
        #         res += 10 + 5 * insideX[1]
        #     return res
        #res
        # mean_cost = predictive_cost(x)

        # print("cost:", mean_cost)
        # mean_cost += 1e-8
        if mean_cost <= 0.0:
            mean_cost = 1e-8
        # mean, _, var = gp.predictScalarLib(x)
        # mean_cost, _, _ = gp_cost.predictScalarLib(x)

        # mean, _, var = gp.predictCholeskyScalar(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        # var[var < 1e-10] = 0
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        result = result/(mean_cost**alpha)
        return result

    @staticmethod
    def _eibs(x, gp, fMax, epsilon, gp_cost, trial):
        mean, _, var = gp.predict(x)
        mean_cost, _, _ = gp_cost.predict(x)
        # mean_cost = gp_cost.costFunction.func(gp_cost.costFunction.denormalize(x), trial)[1]

        if mean_cost <= 0.0:
            mean_cost = 1e-8

        # mean, _, var = gp.predictCholeskyScalar(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        result = result * mean_cost
        return result

    @staticmethod
    def _eicool(x, gp, fMax, epsilon, gp_cost, alpha, trial):
        # print("eicool")
        mean, _, var = gp.predict(x)
        mean_cost, _, _ = gp_cost.predict(x)
        # mean_cost = gp_cost.costFunction.func(gp_cost.costFunction.denormalize(x), trial)[1]
        # print("alpha:",alpha)

        # def predictive_cost(x):
        #     res = 0
        #     # print("x:", x)
        #     # print("shape:", np.shape(x))
        #     insideX = gp_cost.func.denormalize(x)
        #     try:
        #         res += 10 + 5 * insideX[0, 1]
        #         # res += 50 + 15 * np.abs(x[0,0])
        #     except:
        #         res += 10 + 5 * insideX[1]
        #     return res
        # #res
        # mean_cost = predictive_cost(x)

        # print("cost:",mean_cost)
        # mean_cost += 1e-8
        if mean_cost <= 0.0:
            mean_cost = 1e-8
        # mean, _, var = gp.predictScalarLib(x)
        # mean_cost, _, _ = gp_cost.predictScalarLib(x)

        # mean, _, var = gp.predictCholeskyScalar(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        # var[var < 1e-10] = 0
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        # print("alpha:",alpha, " mean_cost:", mean_cost)
        # print("after alpha:",mean_cost**alpha)
        result = result / np.power(mean_cost,alpha)
        return result

    @staticmethod
    def _eibscool(x, gp, fMax, epsilon, gp_cost, alpha, trial):
        # print("eicool")
        mean, _, var = gp.predict(x)
        # mean_cost, _, _ = gp_cost.predict(x)
        mean_cost = gp_cost.costFunction.func(gp_cost.costFunction.denormalize(x), trial)[1]

        if mean_cost <= 0.0:
            mean_cost = 1e-8

        var2 = np.maximum(var, 1e-4 + 0 * var)
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        result = result * np.power(mean_cost, alpha)
        return result

    @staticmethod
    def _ucb(x, gp, beta):
        # if gp.checkExistInObs(x):
        #     return 0
        mean, _, var = gp.predict(x)
        # mean, _, var = gp.predictScalarLib(x)
        # mean, _, var = predictCholeskyScalar(x, Xm, Ym)
        #var[var < 1e-10] = 0
        std = np.sqrt(var)
        result = np.matrix(mean + beta * std)
        return result

    @staticmethod
    def _ucbc(x, gp, beta, gp_cost, trial):
        # mean, _, var = gp.predict(x)
        # mean_cost, _, _ = gp_cost.predict(x)
        mean_cost = gp_cost.costFunction.func(gp_cost.costFunction.denormalize(x), trial)[1]
        if mean_cost <= 0.0:
            mean_cost = 1e-8

        mean, _, var = gp.predict(x)
        std = np.sqrt(var)
        result = np.matrix(mean + beta * std)
        result = mean + 2.0*std/mean_cost #- (mean_cost/np.max(gp_cost.Ym))
        return result

    @staticmethod
    def _ucbps(x, gp, beta, gp_cost, trial):
        # mean, _, var = gp.predict(x)
        # mean_cost, _, _ = gp_cost.predict(x)
        mean_cost = gp_cost.costFunction.func(gp_cost.costFunction.denormalize(x), trial)[1]
        if mean_cost <= 0.0:
            mean_cost = 1e-8

        mean, _, var = gp.predict(x)
        std = np.sqrt(var)
        result = np.matrix(mean + beta * std)
        result = 1.0 * result / mean_cost
        return result

    @staticmethod
    def _lcb(x, gp, beta):
        mean, _, var = gp.predictScalar(x)
        # mean, _, var = predictCholeskyScalar(x, Xm, Ym)
        var[var < 1e-10] = 0
        std = np.sqrt(var)
        result = np.matrix(mean - beta * std)
        return result

    @staticmethod
    def _gaussian(x, mean, sigma=0.1):
        res = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) ** 2) / (sigma ** 2))
        return res



