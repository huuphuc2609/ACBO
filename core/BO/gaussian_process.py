import time
from copy import deepcopy

from scipy.optimize.lbfgsb import _minimize_lbfgsb, fmin_l_bfgs_b
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

import core.GPy
import numpy as np

from core.DIRECT import DIRECTAlgo
from core.GPy import Param
from paramz.transformations import Logexp
import random

from scipy.linalg import lapack
from numpy.linalg import cholesky, det, lstsq, inv

import numdifftools as nd

from core.BO.kernels import Kernels
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial import distance

from core.GPy.kern import Kern, RBF
from core.GPy.kern.src.stationary import Stationary

from sklearn import preprocessing

from core.GPy.util import diag
from core.GPy.util.linalg import pdinv, dpotrs, tdot

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def default_mean_function(x):
    return 0.0

class GaussianProcess():
    def __init__(self,kernelName, numDim, func, isUseLib=True, predictMethod="default", likelihoodCalculation="default", optimizeHyperparams=False, Y_scaler=preprocessing.StandardScaler(), mean_function=default_mean_function):
        self.isUseLib = isUseLib
        if self.isUseLib:
            self.predictMethod = "gpy"
        else:
            # Predict methods: naive, default (cholesky)
            self.predictMethod = predictMethod

        # Likelihood calculation: naive, stable, default (cholesky)
        self.likelihoodCalculation = likelihoodCalculation
        self.optimizeHyperparams = optimizeHyperparams
        self.Y_scaler = Y_scaler
        self.assumedMeanFunction = mean_function

        self.kernel = Kernels(kernelName)
        self.defaultLengthscale = 0.01
        self.defaultAmp = 1.0
        if not self.optimizeHyperparams:
            self.kernel.lengthScale = self.defaultLengthscale
            self.kernel.amp = self.defaultAmp
        self.noise_delta = 1e-8

        self.Xm = []
        self.XmTran = []
        self.Ym = []

        self.kxx = []
        self.kXmXm = []
        self.kXmXm_inv = []
        self.kxXm = []
        self.discrete_idx = []

        self.input_dim = numDim
        self.func = func

        #### If use Lib ####
        if self.isUseLib:
            if kernelName == "rbf":
                self.libKern = core.GPy.kern.RBF(self.input_dim, lengthscale=self.defaultLengthscale,
                                                 ARD=True)  # + GPy.kern.Bias(self.input_dim)
                # self.optimizeLS = True
                # self.libOptimizer = 'bfgs'
            elif kernelName == "linear":
                self.libKern = core.GPy.kern.Linear(self.input_dim, ARD=True)
        self.model = None

    def setCostFunction(self, func):
        self.costFunction = func

    def setCostGP(self, gp):
        self.costGP = gp

    def getLengthScaleLib(self):
        res = self.model.kern.lengthscale.values.tolist()
        return res[0]

    def getAmpLib(self):
        res = self.model.kern.variance.values.tolist()
        return res[0]

    def setLengthScaleLib(self, len):
        # self.libKern = GPy.kern.RBF(self.input_dim, self.func, self.isTran, lengthscale=len, ARD=False)
        self.libKern.lengthscale = Param('lengthscale', len, Logexp())

    def setAmpLib(self, var):
        # self.libKern = GPy.kern.RBF(self.input_dim, self.func, self.isTran, lengthscale=len, ARD=False)
        self.libKern.variance = Param('variance', var, Logexp())

    def dLdT(self, a, iKxx, dKdt):
        """
        Calculates the partial derivatives of the marginal likelihood w.r.t. the hyper-parameters
            dKdt: partial derivative of of the covariance function wrt a hyper-parameter
        """
        return 0.5 * np.trace(np.dot((a @ a.T - iKxx), dKdt))

    def dKdsf(self, x1, x2, var_f, l):
        """Gradient of SE kernel wrt the signal variance s_f"""
        return 2 * np.sqrt(var_f) * np.exp((-distance.cdist(x1, x2) ** 2) / (2 * l ** 2))

    def dKdL(self, x1, x2, var_f, l):
        """Gradient of SE kernel wrt the lengthscale l"""
        return var_f * np.exp((-distance.cdist(x1, x2) ** 2) / (2 * l ** 2)) * (distance.cdist(x1, x2) ** 2) / (l ** 3)

    def Jacobian_of_Log_Marginal_Likelihood(self, x):
        """Calculates the gradients of the objective with respect to signal variance, lengthscale and noise variance"""
        self.kernel.amp = x[1]
        var_f = x[1]
        l = x[0]
        self.kernel.lengthScale = x[0]

        var_n = self.noise_delta * np.eye(len(self.Xm))
        # print("var_n:",var_n)
        Kxx = self.kernel.k(self.Xm, self.Xm)
        Lxx = np.linalg.cholesky(Kxx + var_n)
        a = np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, self.sYm))

        iKxx = np.linalg.inv(Kxx + var_n)

        J = np.empty([2, ])
        J[0] = self.dLdT(a, iKxx, self.dKdL(self.Xm, self.Xm, var_f, l))  # gradient for lengthscale l
        J[1] = self.dLdT(a, iKxx, self.dKdsf(self.Xm, self.Xm, var_f, l))  # gradient for signal variance var_f

        return J

    def fit(self,Xmeasurement,Ymeasurement):
        self.Xm = deepcopy(Xmeasurement)
        self.Ym = deepcopy(Ymeasurement)
        if self.isUseLib:
            inYmeasurement = [[ins] for ins in Ymeasurement]
            self.model = core.GPy.models.GPRegression(np.array(Xmeasurement),np.array(inYmeasurement),self.libKern)
            self.model.optimize_restarts(5, verbose=False)
            return

        # Remove points close to each others
        # filteredXm = []
        # filteredYm = []
        # for idx_insX in range(len(self.Xm)):
        #     isFiltered = False
        #     for insFXm in filteredXm:
        #         # print("insFXm:",insFXm)
        #         # print("self.Xm[idx_insX]:", self.Xm[idx_insX])
        #         dis = np.asscalar(euclidean_distances([insFXm], [self.Xm[idx_insX]]))
        #         # print("dis:",dis)
        #         if dis < 0.2:
        #             isFiltered = True
        #             break
        #     if not isFiltered:
        #         filteredXm.append(deepcopy(self.Xm[idx_insX]))
        #         filteredYm.append(deepcopy(self.Ym[idx_insX]))
        # self.Xm = filteredXm
        # self.Ym = filteredYm

        # Standardize Y to have zero mean
        # self.sYm = preprocessing.scale(self.Ym)
        # self.sYm = (self.Y_scaler.fit_transform((np.array(self.Ym)).reshape(-1,1))).reshape(1,-1)[0]
        self.myMeanY = np.mean(self.Ym)
        self.myStdY = np.std(self.Ym)
        # if self.myStdY != 0:
        #     self.sYm = [(insY-self.myMeanY)/self.myStdY for insY in self.Ym]
        # self.sYm = [insY for insY in self.Ym]
        self.sYm = [(insY - self.myMeanY) / self.myStdY for insY in self.Ym]
        # Estimate the noise using training data (also used by GPy)
        self.y_noise = np.array(self.sYm).var() + self.noise_delta

        if self.isUseLib:
            ########## Use lib ##########
            if self.model is None:
                y_noise = np.array(self.Ym).var() + self.noise_delta
                self.model = core.GPy.models.GPRegression(np.array(self.Xm), np.array(Ymeasurement).reshape(-1,1), self.libKern, noise_var=y_noise, normalizer=True)
                self.model.Gaussian_noise.constrain_fixed(self.noise_delta, warning=False)
            else:
                self.model.set_XY(np.array(self.Xm), np.array(Ymeasurement).reshape(-1,1))
            if self.optimizeHyperparams:
                try:
                    self.model.optimize_restarts(5, verbose=False)
                except np.linalg.linalg.LinAlgError:
                    pass
        else:
            if self.optimizeHyperparams:
                ########## Use own implementation ##########
                if self.kernel.name == "rbf":
                    bnds = ((1e-4, None),(1e-4, None))  # Define boundary tuple

                    optLengthScale = deepcopy(self.kernel.lengthScale)
                    optAmp = deepcopy(self.kernel.amp)
                    if self.likelihoodCalculation == "default":
                        optMLL = self.log_marginal_likelihood()
                    else:
                        if self.likelihoodCalculation == "stable":
                            optMLL = self.nll_stable()
                        if self.likelihoodCalculation == "naive":
                            optMLL = self.nll_naive()


                    candidateThetas = []
                    x0 = [np.float64(self.kernel.lengthScale), np.float64(self.kernel.amp)]
                    upperBoundLS = 2.0
                    lowerBoundLS = 1e-3
                    for num_restart in range(5):
                        opt_result = fmin_l_bfgs_b(func=lambda x: -self.f_log_marginal_likelihood(x), x0=np.array(x0), bounds=[(lowerBoundLS, upperBoundLS), (1e-1, 10.0)], pgtol=1e-37,
                                                     approx_grad=True)
                        # while opt_result[2]['warnflag'] == 2:
                        #     x0 = [np.float64(np.random.uniform(1e-6, 2.0)), np.float64(np.random.uniform(1e-1, 10.0))]
                        #     candidateThetas.append(deepcopy(x0))
                        #
                        #     opt_result = fmin_l_bfgs_b(func=lambda x: -self.f_log_marginal_likelihood(x), x0=np.array(x0),
                        #                                bounds=[(1e-6, 2.0), (1e-1, 10.0)], pgtol=1e-37,
                        #                                approx_grad=True)
                        optimized = opt_result[0]
                        candidateThetas.append(deepcopy(optimized))
                        # x0 = deepcopy(optimized)
                        x0 = [np.float64(np.random.uniform(lowerBoundLS, upperBoundLS)), np.float64(np.random.uniform(1e-1, 2.0))]

                    for cands in candidateThetas:
                        self.kernel.lengthScale = cands[0]
                        self.kernel.amp = cands[1]
                        if self.likelihoodCalculation == "default":
                            curMLL = self.log_marginal_likelihood()
                        else:
                            if self.likelihoodCalculation == "stable":
                                curMLL = self.nll_stable()
                            if self.likelihoodCalculation == "naive":
                                curMLL = self.nll_naive()
                        if curMLL > optMLL:
                            optMLL = curMLL
                            optLengthScale = deepcopy(cands[0])
                            optAmp = deepcopy(cands[1])

                    self.kernel.lengthScale = optLengthScale
                    self.kernel.amp = optAmp

        self.kXmXm = self.kernel.k(self.Xm, self.Xm)
        self.kXmXm = np.add(self.kXmXm, self.noise_delta * np.eye(len(self.Xm)))

    def f_log_marginal_likelihood(self, input):
        self.kernel.lengthScale = input[0]
        self.kernel.amp = input[1]
        if self.likelihoodCalculation == "naive":
            return self.nll_naive()
        if self.likelihoodCalculation == "stable":
            return self.nll_stable()
        else: #default cholesky
            return self.log_marginal_likelihood()

    def f_df_log_marginal_likelihood(self):
        Ymm = deepcopy(np.array(self.sYm))
        K = self.kernel.k(self.Xm, self.Xm)
        Ky = deepcopy(K)
        Ky = np.add(Ky, self.noise_delta * np.eye(len(self.Xm)))
        L = None
        while L is None:
            try:
                L = cholesky(Ky, lower=True)
            except:
                Ky = np.add(Ky, self.noise_delta * np.eye(len(self.Xm)))
                pass
        alpha = cho_solve((L, True), Ymm)
        log_likelihood_dims = -0.5 * np.float64(np.sum(Ymm * alpha))
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= 0.5 * np.float64(len(self.Xm))* np.log(2.0 * np.pi)
        log_likelihood = np.float64(log_likelihood_dims.sum(-1))
        ### Correct one ###

        """Calculates the gradients of the objective with respect to signal variance, lengthscale and noise variance"""
        var_f = self.kernel.amp
        l = self.kernel.lengthScale

        var_n = self.noise_delta * np.eye(len(self.Xm))
        # print("var_n:",var_n)
        Kxx = deepcopy(K)
        a = alpha
        # Lxx = np.linalg.cholesky(Kxx + var_n)
        # a = np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, self.sYm))


        iKxx = np.linalg.inv(Kxx + var_n)

        J = np.empty([2, ])
        J[0] = self.dLdT(a, iKxx, self.dKdL(self.Xm, self.Xm, var_f, l))  # gradient for lengthscale l
        J[1] = self.dLdT(a, iKxx, self.dKdsf(self.Xm, self.Xm, var_f, l))  # gradient for signal variance var_f

        return log_likelihood, J

    def log_marginal_likelihood(self):
        ### Cholesky ####
        Ymm = deepcopy(np.array(self.sYm))
        K = self.kernel.k(self.Xm, self.Xm)
        Ky = deepcopy(K)
        Ky = np.add(Ky, self.noise_delta * np.eye(len(self.Xm)))
        L = None
        # while L is None:
        #     try:
        #         L = cholesky(Ky, lower=True)
        #     except:
        #         Ky = np.add(Ky, self.noise_delta * np.eye(len(self.Xm)))
        #         pass
        # if not is_pos_def(Ky):
        #     return -9999
        if not is_pos_def(Ky):
            np.set_printoptions(precision=3)
            # print("Xm:", self.Xm)
            # print("Ky:", Ky)
            # print("lengthscale:", self.kernel.lengthScale, " amp:", self.kernel.amp)
        try:
            L = cholesky(Ky, lower=True)
        except:
            return -100

        # L = cholesky(Ky, lower=True)
        alpha = cho_solve((L, True), Ymm)
        # print("noise:", self.noise_delta, " sum K:", np.sum(K), " alpha * Y my:", np.sum(Ymm * alpha), " W:", np.log(np.diag(L)).sum())
        # log_likelihood_dims = -0.5 * np.sum(np.dot(Ymm, alpha))
        ### Correct one ###
        log_likelihood_dims = -0.5 * np.float64(np.sum(Ymm * alpha))
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= 0.5 * np.float64(len(self.Xm))* np.log(2.0 * np.pi)
        log_likelihood = np.float64(log_likelihood_dims.sum(-1))

        ### Simplified version ###
        # log_likelihood_dims = -0.5 * np.float64(np.sum(Ymm * alpha))
        # log_likelihood_dims -= np.log(np.diag(L)).sum()
        # # log_likelihood_dims -= 0.5 * np.float64(len(self.Xm))* np.log(2.0 * np.pi)
        # log_likelihood = np.float64(log_likelihood_dims.sum(-1))

        #### GPyLob ####
        # YYT_factor = (np.array(self.sYm)).copy()
        # K = np.add(self.kernel.k(self.Xm, self.Xm), self.noise_delta * np.eye(len(self.Xm)))
        # Ky = K.copy()
        # LW = None
        # try:
        #     Wi, LW, LWi, W_logdet = pdinv(Ky)
        # except:
        #     return -9999
        # # Wi, LW, LWi, W_logdet = pdinv(Ky)
        # # while LW is None:
        # #     try:
        # #         Wi, LW, LWi, W_logdet = pdinv(Ky)
        # #     except:
        # #         Ky = np.add(Ky, self.noise_delta * np.eye(len(self.Xm)))
        # #         pass
        #
        # alpha, _ = dpotrs(LW, YYT_factor, lower=1)
        #
        # log_likelihood = 0.5*(-len(self.Xm) * np.log(2.0 * np.pi) - W_logdet - np.sum(alpha * YYT_factor))

        # print("my alpha:", np.sum(alpha), " yyt:", np.sum(YYT_factor), "K:", np.sum(self.kernel.k(self.Xm, self.Xm)), " Ky:", np.sum(Ky) , " log:", log_likelihood)
        # print("log:", log_likelihood)
        # # dL_dK = 0.5 * (tdot(alpha) - Wi)
        #
        # # dL_dthetaL = (np.diag(dL_dK)).sum()

        # print("dL_dthetaL:", dL_dthetaL)
        return np.float64(log_likelihood)

    def nll_naive(self):
        # Naive implementation of Eq. (7). Works well for the examples
        # in this article but is numerically less stable compared to
        # the implementation in nll_stable below.
        K = np.add(self.kernel.k(self.Xm, self.Xm), self.noise_delta * np.eye(len(self.Xm)))
        Ymm = np.array(self.sYm)
        return -1.0 * (0.5 * np.log(det(K)) + \
               0.5 * Ymm.T.dot(inv(K).dot(Ymm)) + \
               0.5 * len(self.Xm) * np.log(2.0*np.pi))

    def nll_stable(self):
        # Numerically more stable implementation of Eq. (7) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        # K = self.kernel.k(self.Xm, self.Xm) + self.noise_delta * np.eye(len(self.Xm))
        K = np.add(self.kernel.k(self.Xm, self.Xm), self.noise_delta * np.eye(len(self.Xm)))
        L = cholesky(K)
        Ymm = np.array(self.sYm)

        res = -1.0 * (np.sum(np.log(np.diagonal(L))) + \
               0.5 * Ymm.T.dot(lstsq(L.T, lstsq(L, Ymm, rcond=None)[0], rcond=None)[0]) + \
               0.5 * len(self.Xm) * np.log(2.0*np.pi))

        return res

    def predict(self,x):
        if self.isUseLib:
            mean, var = self.model.predict(np.array([x]))
            return np.asscalar(mean), None, np.asscalar(var)
        if self.predictMethod == "gpy":
            return self.predictScalarLib(x)
        else:
            if self.predictMethod == "default":
                # print("default")
                return self.predictCholeskyScalar(x)
            if self.predictMethod == "naive":
                return self.predictScalar(x)

    def predictScalar(self, x):
        x = np.array(x).reshape(1, -1)
        # print("x:",x)
        # print("my var", np.array(self.Xm).var())
        self.kxx = self.kernel.k(x, x)
        # print("kxx:", self.kxx)
        ### Already calculated after fitting ###
        # self.kXmXm = self.kernel.k(self.Xm, self.Xm)
        # self.kXmXm = np.add(self.kXmXm, 1e-8 * np.eye(len(self.Xm)))
        ########################################
        # print("self.kXmXm:",self.kXmXm)
        self.kXmXm_inv = np.linalg.pinv(self.kXmXm)

        self.kxXm = self.kernel.k(x, self.Xm)

        kxXm_XmXmInv = np.dot(self.kxXm, self.kXmXm_inv)

        mu_s = np.dot(kxXm_XmXmInv, self.sYm)
        # cov_s = self.kxx - np.sum(np.dot(kxXm_XmXmInv, np.transpose(self.kxXm)))
        cov_s = self.kxx - np.sum(np.dot(kxXm_XmXmInv, self.kxXm.T))
        var_s = np.matrix(np.diag(cov_s))
        var_s = np.clip(var_s, 1e-15, np.inf)

        # inverse_mu = (mu_s * np.std(self.Ym)) + np.mean(self.Ym)
        # inverse_var = var_s * (np.std(self.Ym)**2)
        inverse_mu = self.Y_scaler.inverse_transform(mu_s)
        inverse_var = self.Y_scaler.var_ * var_s
        # inverse_var = var_s*(self.Ystd**2)

        return inverse_mu, cov_s, inverse_var

    def predictCholeskyScalar(self, x):
        # self.ThisCall+=1
        # start1 = time.time()
        x = np.array(x).reshape(1, -1)
        ### Already calculated after fitting ###
        # self.kXmXm = self.kernel.k(self.Xm, self.Xm)
        # self.kXmXm = np.add(self.kXmXm, 1e-8 * np.eye(len(self.Xm)))
        ########################################
        self.kxXm = self.kernel.k(x, self.Xm)

        try:
            L = cholesky(self.kXmXm, lower=True)
        except:
            # print("self.kXmXm:", self.kXmXm)
            # print("len: ", self.kernel.lengthScale, " amp:", self.kernel.amp)
            # print("is pos def:", is_pos_def(self.kXmXm))
            posK = np.clip(self.kXmXm, 1e-15, np.inf)
            try:
                L = cholesky(posK, lower=True)
            except:
                posK = nearestPD(posK)
                L = cholesky(posK, lower=True)

        # print("self.sYm:",self.sYm)
        alpha = cho_solve((L, True), self.sYm)
        # y_train_mean = np.mean(Y_query)
        mu_s = self.kxXm.dot(alpha)  # Line 4 (y_mean = f_star)
        # y_mean = y_train_mean + y_mean  # undo normal.

        ######################CovByCholesky######################
        # v = cho_solve((L, True), self.kxXm.T)  # Line 5
        # cov_s = self.kernel.k(x, x) - self.kxXm.dot(v)  # Line 6
        # stop1 = time.time()

        # start2 = time.time()
        ######################StandardDeviationByCholesky######################
        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        # L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
        # self.kXmXm_inv = L_inv.dot(L_inv.T)
        # Compute variance of predictive distribution
        # y_var = np.matrix(np.diag(np.matrix(self.kernel.k(x, x)))) - np.einsum("ij,ij->i", np.dot(self.kxXm, self.kXmXm_inv),
        #                                                                        self.kxXm)
        # y_var = np.clip(y_var, 1e-15, np.inf)
        # inverse_mu = self.Y_scaler.inverse_transform(mu_s)
        # inverse_mu = self.Y_scaler.inverse_transform(mu_s)
        # inverse_mu = mu_s * np.sqrt(self.Y_scaler.var_) + self.Y_scaler.mean_
        inverse_mu = mu_s*self.myStdY + self.myMeanY
        # inverse_var = self.Y_scaler.var_ * y_var
        # inverse_var = y_var * self.myStdY**2
        # std_s = np.sqrt(np.absolute(y_var))
        # stop2 = time.time()
        # print("time 1:", stop1 - start1)
        # print("time 2:", stop2 - start2)
        # return inverse_mu, cov_s, inverse_var
        return inverse_mu, None, 0.0

    def predictScalarDebug(self, x):
        x = np.array(x).reshape(1, -1)
        # print("x:",x)
        # print("my var", np.array(self.Xm).var())
        self.kxx = self.kernel.k(x, x)
        # print("kxx:", self.kxx)
        self.kXmXm = self.kernel.k(self.Xm, self.Xm)
        print("self.kXmXm:",self.kXmXm)
        self.kXmXm_inv = np.linalg.pinv(self.kXmXm)

        self.kxXm = self.kernel.k(x, self.Xm)

        kxXm_XmXmInv = np.dot(self.kxXm, self.kXmXm_inv)

        mu_s = np.dot(kxXm_XmXmInv, self.sYm)
        # cov_s = self.kxx - np.sum(np.dot(kxXm_XmXmInv, np.transpose(self.kxXm)))
        cov_s = self.kxx - np.sum(np.dot(kxXm_XmXmInv, self.kxXm.T))
        var_s = np.matrix(np.diag(cov_s))
        var_s = np.clip(var_s, 1e-15, np.inf)

        # inverse_mu = (mu_s * np.std(self.Ym)) + np.mean(self.Ym)
        # inverse_var = var_s * (np.std(self.Ym)**2)
        inverse_mu = self.Y_scaler.inverse_transform(mu_s)
        inverse_var = self.Y_scaler.var_ * var_s

        return inverse_mu, cov_s, inverse_var

    def predictScalarLib(self, x):
        x = np.array(x).reshape(1, -1)
        #use lib
        #newModel = self.model.copy()
        # m, v = newModel.predict(x, full_cov=False, include_likelihood=False)
        m, v = self.model.predict(x, full_cov=False, include_likelihood=False)

        var_s = np.clip(v, 1e-15, np.inf)
        mu_s = m

        return mu_s, 0, var_s

class CostGaussianProcess(GaussianProcess):
    def __init__(self,kernelName, numDim, func, trial, isUseLib=True, predictMethod="default", likelihoodCalculation="default", optimizeHyperparams=False, Y_scaler=preprocessing.StandardScaler()):
        super().__init__(kernelName, numDim, func, isUseLib, predictMethod, likelihoodCalculation, optimizeHyperparams, Y_scaler)

        if kernelName == "cost_rbf":
            self.libKern = core.GPy.kern.CostRBF(self.input_dim, lengthscale=self.defaultLengthscale,
                                             ARD=True)  # + GPy.kern.Bias(self.input_dim)
            self.optimizeLS = True
            self.libOptimizer = 'bfgs'
        else:
            self.libKern = core.GPy.kern.RBF(self.input_dim, lengthscale=self.defaultLengthscale,
                                             ARD=True)  # + GPy.kern.Bias(self.input_dim)
            self.optimizeLS = True
            self.libOptimizer = 'bfgs'

        self.trial = trial

    def log_marginal_likelihood(self):
        ### Cholesky ####
        Ymm = deepcopy(np.array(self.sYm))
        K = self.kernel.k(self.Xm, self.Xm, self.costXm, self.costXm)
        Ky = deepcopy(K)
        Ky = np.add(Ky, self.noise_delta * np.eye(len(self.Xm)))
        L = None
        # while L is None:
        #     try:
        #         L = cholesky(Ky, lower=True)
        #     except:
        #         Ky = np.add(Ky, self.noise_delta * np.eye(len(self.Xm)))
        #         pass
        # if not is_pos_def(Ky):
        #     return -9999
        # print("Ky:", Ky)
        # if not is_pos_def(Ky):
            # np.set_printoptions(precision=3)
            # print("Xm:", self.Xm)
            # print("Ky:", Ky)
            # print("lengthscale:", self.kernel.lengthScale, " amp:", self.kernel.amp)
        pen = 0.0
        try:
            L = cholesky(Ky, lower=True)
        except:
            Ky = nearestPD(Ky)
            L = cholesky(Ky, lower=True)
            pen += 2.0
            # return -99999

        L = cholesky(Ky, lower=True)
        alpha = cho_solve((L, True), Ymm)

        ### Correct one ###
        log_likelihood_dims = -0.5 * np.float64(np.sum(Ymm * alpha))
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= 0.5 * np.float64(len(self.Xm)) * np.log(2.0 * np.pi)
        log_likelihood = np.float64(log_likelihood_dims.sum(-1))

        return np.float64(log_likelihood) - pen

    def fit(self,Xmeasurement,Ymeasurement):
        self.Xm = deepcopy(Xmeasurement)
        self.Ym = deepcopy(Ymeasurement)
        self.costXm = self.costGP.Ym
        # Remove points close to each others
        # filteredXm = []
        # filteredYm = []
        # for idx_insX in range(len(self.Xm)):
        #     isFiltered = False
        #     for insFXm in filteredXm:
        #         # print("insFXm:",insFXm)
        #         # print("self.Xm[idx_insX]:", self.Xm[idx_insX])
        #         dis = np.asscalar(euclidean_distances([insFXm], [self.Xm[idx_insX]]))
        #         # print("dis:",dis)
        #         if dis < 0.2:
        #             isFiltered = True
        #             break
        #     if not isFiltered:
        #         filteredXm.append(deepcopy(self.Xm[idx_insX]))
        #         filteredYm.append(deepcopy(self.Ym[idx_insX]))
        # self.Xm = filteredXm
        # self.Ym = filteredYm

        # Standardize Y to have zero mean
        # self.sYm = preprocessing.scale(self.Ym)
        # self.sYm = (self.Y_scaler.fit_transform((np.array(self.Ym)).reshape(-1,1))).reshape(1,-1)[0]
        self.myMeanY = np.mean(self.Ym)
        self.myStdY = np.std(self.Ym)
        self.sYm = [(insY-self.myMeanY)/self.myStdY for insY in self.Ym]
        # Estimate the noise using training data (also used by GPy)
        self.y_noise = np.array(self.sYm).var() + self.noise_delta

        if self.isUseLib:
            ########## Use lib ##########
            if self.model is None:
                y_noise = np.array(self.Ym).var() + self.noise_delta
                self.model = core.GPy.models.GPRegression(np.array(self.Xm), np.array(Ymeasurement).reshape(-1,1), self.libKern, noise_var=y_noise, normalizer=True)
                self.model.Gaussian_noise.constrain_fixed(self.noise_delta, warning=False)
            else:
                self.model.set_XY(np.array(self.Xm), np.array(Ymeasurement).reshape(-1,1))
            if self.optimizeHyperparams:
                try:
                    self.model.optimize_restarts(5, verbose=False)
                except np.linalg.linalg.LinAlgError:
                    pass
        else:
            ########## Use own implementation ##########
            if self.optimizeHyperparams:
                if self.kernel.name == "rbf" or self.kernel.name == "cost_rbf" or self.kernel.name == "cost_rbf_opt":
                    bnds = ((1e-4, None),(1e-4, None))  # Define boundary tuple

                    optLengthScale = deepcopy(self.kernel.lengthScale)
                    optAmp = deepcopy(self.kernel.amp)
                    if self.likelihoodCalculation == "default":
                        optMLL = self.log_marginal_likelihood()
                    else:
                        if self.likelihoodCalculation == "stable":
                            optMLL = self.nll_stable()
                        if self.likelihoodCalculation == "naive":
                            optMLL = self.nll_naive()


                    candidateThetas = []
                    x0 = [np.float64(self.kernel.lengthScale), np.float64(self.kernel.amp)]
                    upperBoundLS = 2.0
                    lowerBoundLS = 1e-4
                    for num_restart in range(5):
                        opt_result = fmin_l_bfgs_b(func=lambda x: -self.f_log_marginal_likelihood(x), x0=np.array(x0), bounds=[(lowerBoundLS, upperBoundLS), (1e-1, 10.0)], pgtol=1e-37,
                                                     approx_grad=True)
                        # while opt_result[2]['warnflag'] == 2:
                        #     x0 = [np.float64(np.random.uniform(1e-6, upperBoundLS)), np.float64(np.random.uniform(1e-1, 10.0))]
                        #     candidateThetas.append(deepcopy(x0))
                        #
                        #     opt_result = fmin_l_bfgs_b(func=lambda x: -self.f_log_marginal_likelihood(x), x0=np.array(x0),
                        #                                bounds=[(1e-6, upperBoundLS), (1e-1, 10.0)], pgtol=1e-37,
                        #                                approx_grad=True)
                        optimized = opt_result[0]
                        candidateThetas.append(deepcopy(optimized))
                        # x0 = deepcopy(optimized)
                        x0 = [np.float64(np.random.uniform(lowerBoundLS, upperBoundLS)), np.float64(np.random.uniform(1e-1, 2.0))]

                    for cands in candidateThetas:
                        self.kernel.lengthScale = cands[0]
                        self.kernel.amp = cands[1]
                        if self.likelihoodCalculation == "default":
                            curMLL = self.log_marginal_likelihood()
                        else:
                            if self.likelihoodCalculation == "stable":
                                curMLL = self.nll_stable()
                            if self.likelihoodCalculation == "naive":
                                curMLL = self.nll_naive()
                        if curMLL > optMLL:
                            optMLL = curMLL
                            optLengthScale = deepcopy(cands[0])
                            optAmp = deepcopy(cands[1])

                    self.kernel.lengthScale = optLengthScale
                    self.kernel.amp = optAmp

        self.kXmXm = self.kernel.k(self.Xm, self.Xm, self.costXm, self.costXm)
        self.kXmXm = np.add(self.kXmXm, self.noise_delta * np.eye(len(self.Xm)))

        try:
            self.L = cholesky(self.kXmXm, lower=True)
        except:
            # print("self.kXmXm:", self.kXmXm)
            # print("len: ", self.kernel.lengthScale, " amp:", self.kernel.amp)
            # print("is pos def:", is_pos_def(self.kXmXm))
            posK = np.clip(self.kXmXm, 1e-15, np.inf)
            try:
                self.L = cholesky(posK, lower=True)
            except:
                try:
                    posK = nearestPD(posK)
                    posK = np.clip(posK, 1e-15, np.inf)
                    posK = np.add(posK, 1e-3 * np.eye(len(posK)))
                    self.L = cholesky(posK, lower=True)
                except:
                    posK = np.clip(posK, 1e-15, np.inf)
                    posK = np.add(posK, 1e-3 * np.eye(len(posK)))
                    self.L = cholesky(posK, lower=True)

        self.alpha = cho_solve((self.L, True), self.sYm)

    def predict(self,x):
        if self.isUseLib:
            mean, var = self.model.predict(np.array([x]))
            return np.asscalar(mean), None, np.asscalar(var)
        if self.predictMethod == "gpy":
            return self.predictScalarLib(x)
        else:
            if self.predictMethod == "default":
                # print("default")
                return self.predictCholeskyScalar(x)
            if self.predictMethod == "naive":
                return self.predictScalar(x)

    def predictCholeskyScalar(self, x):
        x = np.array(x).reshape(1, -1)
        ### Already calculate after fitting ###
        # self.kXmXm = self.kernel.k(self.Xm, self.Xm, self.costXm, self.costXm)
        # self.kXmXm = np.add(self.kXmXm, 1e-8 * np.eye(len(self.Xm)))
        #######################################
        # costx = [self.costGP.predict(ins)[0] for ins in x] # Currently used for real exp
        costx = [np.asscalar(self.costGP.costFunction.func(self.costGP.costFunction.denormalize(ins), self.trial)[1]) for ins in x] # for synthetic
        # costx = [np.asscalar(self.scaler.transform((self.costGP.predict(ins)[0]).reshape(-1,1))) for ins in x]
        # print("costx:",costx)
        self.kxXm = self.kernel.k(x, self.Xm, costx, self.costXm)

        self.kxkx = self.kernel.k(x, x, costx, costx)

        # try:
        #     L = cholesky(self.kXmXm, lower=True)
        # except:
        #     # print("self.kXmXm:", self.kXmXm)
        #     # print("len: ", self.kernel.lengthScale, " amp:", self.kernel.amp)
        #     # print("is pos def:", is_pos_def(self.kXmXm))
        #     posK = np.clip(self.kXmXm, 1e-15, np.inf)
        #     try:
        #         L = cholesky(posK, lower=True)
        #     except:
        #         posK = nearestPD(posK)
        #         posK = np.clip(posK, 1e-15, np.inf)
        #         posK = np.add(posK, 1e-3 * np.eye(len(posK)))
        #         L = cholesky(posK, lower=True)
        #
        # alpha = cho_solve((L, True), self.sYm)
        # # y_train_mean = np.mean(Y_query)
        # mu_s = self.kxXm.dot(alpha)  # Line 4 (y_mean = f_star)
        # # y_mean = y_train_mean + y_mean  # undo normal.
        mu_s = self.kxXm.dot(self.alpha)
        ######################CovByCholesky######################
        # try:
        #     v = cho_solve((L, True), self.kxXm.T)  # Line 5
        # except:
        #     print("alpha:",self.kernel.alpha)
        # print("L:", L)

        # v = cho_solve((L, True), self.kxXm.T)  # Line 5
        # cov_s = self.kxkx - self.kxXm.dot(v)  # Line 6

        ######################StandardDeviationByCholesky######################
        # # compute inverse K_inv of K based on its Cholesky
        # # decomposition L and its inverse L_inv
        # L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
        # self.kXmXm_inv = L_inv.dot(L_inv.T)
        # # Compute variance of predictive distribution
        # y_var = np.matrix(np.diag(np.matrix(self.kxkx))) - np.einsum("ij,ij->i", np.dot(self.kxXm, self.kXmXm_inv),
        #                                                                        self.kxXm)
        # y_var = np.clip(y_var, 1e-15, np.inf)
        # inverse_mu = mu_s * self.myStdY + self.myMeanY
        # inverse_var = y_var * self.myStdY ** 2
        # # std_s = np.sqrt(np.absolute(y_var))
        # # return inverse_mu, cov_s, inverse_var

        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))
        self.kXmXm_inv = L_inv.dot(L_inv.T)
        # Compute variance of predictive distribution
        y_var = np.matrix(np.diag(np.matrix(self.kxkx))) - np.einsum("ij,ij->i", np.dot(self.kxXm, self.kXmXm_inv),
                                                                     self.kxXm)
        y_var = np.clip(y_var, 1e-15, np.inf)
        inverse_mu = mu_s * self.myStdY + self.myMeanY
        inverse_var = y_var * self.myStdY ** 2
        # std_s = np.sqrt(np.absolute(y_var))
        # return inverse_mu, cov_s, inverse_var

        return inverse_mu, None, inverse_var

class CostKernel(Kern):
    def __init__(self, input_dim, variance=1., lengthscale=0.01, ARD=True, active_dims=None, name='costrbf', useGPU=False, inv_l=False):
        super(CostKernel, self).__init__(input_dim, active_dims, name, useGPU=useGPU)
        self.variance = Param('variance', variance)
        self.lengthscale = Param('lengtscale', lengthscale)
        self.link_parameters(self.variance, self.lengthscale)

    def K(self, X, X2):
        # if X2 is None: X2 = X
        # dist2 = np.square((X - X2.T) / self.lengthscale)
        # return self.variance * (1 + dist2 / 2.) ** (-self.power)

        # alpha = (totalBud - usedBud) / (totalBud - initBud)

        def predictive_cost(x):
            res = 0
            # print("x:", x)
            # print("shape:", np.shape(x))
            try:
                res += 2 + 5 * x[0, 1]
            except:
                res += 2 + 5 * x[1]

            return res

        def cost_euclidean_distance(x, xprime):
            # x = (x.tolist())[0]
            # xprime = (xprime.tolist())[0]
            # print("x:", x)
            # print("xprime:", xprime)
            cost_x = predictive_cost(x)
            cost_xprime = predictive_cost(xprime)
            res = cost_xprime - cost_x
            return res

        warpedX = []
        warpedXprime = []
        if X is not None:
            for i in range(np.shape(X)[0]):
                warpedX.append([predictive_cost(X[i])])
        if X2 is not None:
            for i in range(np.shape(X2)[0]):
                warpedXprime.append([predictive_cost(X2[i])])
        print("here")
        print("warpedX:",warpedX)
        print("warpedXprime:", warpedXprime)

        # if X2 is not None:
        #     r = cost_euclidean_distance(X, X2) + 1e-8
        # else:
        #     r = 1
        r = 0

        r2 = euclidean_distances(X, X2) + r
        result = self.variance * np.exp(-0.5 * r2 ** 2 / self.lengthscale ** 2)
        return result

    def Kdiag(self, X):
        return self.variance * np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X
        dist2 = -0.5 * euclidean_distances(X, X2) ** 2 / self.lengthscale ** 2

        dvar = 2.0*self.variance*dist2
        dl = self.variance * dist2 * (-2) * self.lengthscale * dist2

        self.variance.gradient = np.sum(dvar * dL_dK)
        self.lengthscale.gradient = np.sum(dl * dL_dK)
    #
    # def update_gradients_diag(self, dL_dKdiag, X):
    #     self.variance.gradient = np.sum(dL_dKdiag)
    #     # here self.lengthscale and self.power have no influence on Kdiag so target[1:] are unchanged