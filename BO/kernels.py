from copy import deepcopy

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.gaussian_process.kernels import Matern as skMatern

# import warnings
# warnings.filterwarnings("ignore")

class Kernels:
    def __init__(self,inputName):
        self.name = inputName
        self.lengthScale = 0.01
        self.amp = 1.0

    def setCostGP(self, gp):
        self.costGP = gp

    def setAlpha(self, val):
        self.alpha = val

    def setTurn(self, val):
        self.turn = val

    def k(self, x, xprime,costx=None,costxprime=None):
        if self.name == 'matern':
            return self._mattern_kernel(x,xprime,self.lengthScale, self.amp)
        if self.name == 'ise':
            return self._ise_kernel(x,xprime,self.lengthScale, self.amp)
        if self.name == 'rbf':
            return self._radial_basis_kernel(x,xprime,self.lengthScale,self.amp)
        if self.name == 'mixed':
            return self._mixed_basis_kernel(x,xprime,self.lengthScale,self.amp)
        if self.name == "cost_rbf":
            return self._radial_basis_cost_kernel(x,xprime,costx,costxprime,self.lengthScale,self.amp, costgp=self.costGP, alpha=self.alpha, turn=self.turn)
        if self.name == "cost_rbf_opt":
            return self._radial_basis_cost_kernel_opt(x,xprime,costx,costxprime,self.lengthScale,self.amp, costgp=self.costGP, alpha=self.alpha)
        if self.name == "linear":
            return self._linear_kernel(x, xprime, self.lengthScale, self.amp)

    @staticmethod
    def _linear_kernel(x, xprime, len, amp):
        x = np.array(x)
        xprime = np.array(xprime)
        c = np.ones(x.shape) * len
        cprime = np.ones(xprime.shape) * len

        result = (amp) * np.dot((x-c),(xprime - cprime).T)
        # result = (amp) * np.dot(x, xprime.T)
        # result = np.sqrt(amp) * np.exp((-distance.cdist(x, xprime) ** 2) / (2 * len ** 2))
        # result = (amp) * np.exp(-0.5 * (euclidean_distances(x, xprime, squared=True)) / (len ** 2))
        # result = (amp) * np.exp(-0.5 * (pairwise_distances(x, xprime, metric="minkowski")**2) / (len ** 2))
        return result

    @staticmethod
    def _radial_basis_kernel(x, xprime, len, amp):
        x = np.array(x)
        xprime = np.array(xprime)
        sqdist = np.sum(x ** 2, 1).reshape(-1, 1) + np.sum(xprime ** 2, 1) - 2.0 * np.dot(x, xprime.T)
        result = (amp) * np.exp(-0.5 * sqdist / (len**2))
        # result = np.sqrt(amp) * np.exp((-distance.cdist(x, xprime) ** 2) / (2 * len ** 2))
        # result = (amp) * np.exp(-0.5 * (euclidean_distances(x, xprime, squared=True)) / (len ** 2))
        # result = (amp) * np.exp(-0.5 * (pairwise_distances(x, xprime, metric="minkowski")**2) / (len ** 2))
        return result

    def _drbfDLen(self, x, xprime, len, amp):
        # Derivative of rbf w.r.t amp
        sqdist = np.sum(x ** 2, 1).reshape(-1, 1) + np.sum(xprime ** 2, 1) - 2.0 * np.dot(x, xprime.T)
        return 2.0*np.sqrt(amp) * np.exp(-0.5 * sqdist / (len**2))

    def _drbfDAmp(self, x, xprime, len, amp):
        sqdist = np.sum(x ** 2, 1).reshape(-1, 1) + np.sum(xprime ** 2, 1) - 2.0 * np.dot(x, xprime.T)
        res = (amp) * np.exp(-0.5 * sqdist / (len**2)) * ((sqdist) / (len**3))
        return res

    @staticmethod
    def _mattern_kernel(x, xprime, lengthScale, amp):
        x = np.array(x)
        xprime = np.array(xprime)
        K = skMatern(nu=amp, length_scale=lengthScale)
        return K(x, xprime)

    @staticmethod
    def _ise_kernel(x, xprime, lengthScale, amp=1.0):
        ''' Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2.
        Args:   X1: Array of m points (m x d).
                X2: Array of n points (n x d).
        Returns: Covariance matrix (m x n). '''
        x = np.array(x)
        xprime = np.array(xprime)
        #sqdist = np.sum(x ** 2, 1).reshape(-1, 1) + np.sum(xprime ** 2, 1) - 2 * np.dot(x, xprime.T)
        sqdist = euclidean_distances(x,xprime)**2
        # K = amp * np.exp(-0.5 / (lengthScale ** 2) * sqdist)
        result = np.exp(-0.5 * sqdist) / (lengthScale ** 2)
        return result

    @staticmethod
    def _radial_basis_cost_kernel(x, xprime, costx, costxprime, len, amp, costgp, alpha, turn):
        # alpha = (totalBud - usedBud) / (totalBud - initBud)
        # alpha = 0.0
        # alpha = 0.01
        # print("alpha:",alpha)
        in_x = np.array(deepcopy(x))
        in_xprime = np.array(deepcopy(xprime))

        #### Current kernels following ref ####
        noise_y = 0.1
        rcost = []
        # rr = []
        # Kronecker = []
        for idx1 in range(np.shape(in_x)[0]):
            row = []
            # rr_row = []
            # r1 = costx[idx1]
            r1 = costx[idx1]
            # r1 += 1.0
            r1 = r1 / np.max(costgp.Ym) + 1.0
            # r1 = r1 / np.max(costgp.Ym) * len + 1.0
            # r1 = np.log(r1)
            # r1 = r1 / np.max(costgp.Ym) + 1e-6
            # r1 = (r1 / 2.0) / np.max(costgp.Ym) + 1e-6

            # r1 = r1 / np.max(costgp.Ym)
            # r1 = r1
            # r1 = r1/np.sum(costgp.Ym)
            # if turn % 2 == 0:
            #     r1 = np.abs(1.0 - r1)
                # if r1 < 1.0:
                #     r1 += 1.0

            for idx2 in range(np.shape(in_xprime)[0]):
                # r2 = costxprime[idx2]
                r2 = costxprime[idx2]
                # r2 += 1.0
                # r2 = r2*len
                r2 = r2 / np.max(costgp.Ym) + 1.0
                # r2 = r2 / np.max(costgp.Ym) * len + 1.0
                # r2 = np.log(r2)
                # r2 = r2 / np.max(costgp.Ym) + 1e-6
                # r2 = (r2/2.0) / np.max(costgp.Ym) + 1e-6

                # r2 = r2 / np.max(costgp.Ym)
                # r2 = r2
                # r2 = r2/np.sum(costgp.Ym)
                # r1_r2 = ((np.abs(r1+r2) ** alpha))**2
                # r1_r2 = ((np.abs(r1 + r2)) ** 2)**alpha
                # r1_r2 = ((np.abs(r1**2 + r2**2) ** alpha))

                # if turn % 2 == 0:
                #     r2 = np.abs(1.0 - r2)
                    # if r2 < 1.0:
                    #     r2 += 1.0

                # r1_r2 = (((r1 ** 2 + r2 ** 2)) ** alpha)#Branin_c3 good * 0.001
                r1_r2 = (np.abs((r1 ** 2) - (r1 ** 2)) ** alpha)  # Branin_c3 good * 0.001
                # r1_r2 = ((r1 ** 2 + r2 ** 2))  # Branin_c3 good * 0.001

                # r1_r2 = (((r1 ** 2 + r2 ** 2)) ** alpha) * 100.0 # Branin_c3 good
                # r1_r2 = (((r1*r2)) ** alpha)
                # r1_r2 = (((r1 ** 2 + r2 ** 2)) ** alpha)
                # r1_r2 = ((r1*len) ** 2 + (r2*len) ** 2) ** alpha
                r1_r2 = np.asscalar(r1_r2)
                # r1_r2 = np.clip(r1_r2, 1e-6, np.inf)
                # print("r1_r2:",r1_r2)
                # if r1_r2 > 0:
                #     r1_r2 *= len
                #     r1_r2 = r1_r2**2
                # else:
                #     r1_r2 = len**2
                row.append(r1_r2)
                # r1r2 = (r1*r2*len*len) ** alpha
                # rr_row.append(2.0*r1r2)
                # if (np.linalg.norm(in_x[idx1] - in_xprime[idx2]) == 0):
                #     Kron_row.append(1.0*noise_y)
                # else:
                #     Kron_row.append(0.0)
            rcost.append(row)
            # rr.append(rr_row)
            # Kronecker.append(Kron_row)
        rcost = np.array(rcost)
        # rr = np.array(rr)
        # rcost = costgp.Y_scaler.transform(rcost)
        rcost = np.add(rcost, 1e-8 * np.ones(rcost.shape))
        # rr = np.add(rr, 1e-8 * np.ones(rr.shape))
        r = euclidean_distances(in_x, in_xprime) ** 2

        # warped_r = np.multiply(r, rcost)
        # warped_r = np.add(r,rcost) # Good
        # if turn%2 == 0:
        #     warped_r = np.divide(r, rcost)
        # else:
        #     warped_r = r
        warped_r = np.divide(r, rcost)
        # np.set_printoptions(precision=3)
        # print("r:", r)
        # print("rcost:", rcost)
        # print("warped_r:", warped_r)
        # exit()
        # amp = amp
        # amp = amp / (np.sqrt(rcost) * len)
        # if turn % 2 == 0:
        # res = amp * np.exp(-0.5 * warped_r / (len ** 2))
        # else:

        # res = amp * np.exp(-0.5 * warped_r / (len ** 2))
        rcost_outer = rcost * len**2
        # rcost_outer = np.sqrt(rcost_outer)
        res = amp * np.exp(-0.5 * warped_r / (len ** 2)) / rcost_outer # Current using
        # res = amp * np.exp(-0.5 * warped_r/(len**2))

        # res = amp * np.exp(-0.5 * r / (len ** 2)) / rcost

        # res = amp * np.exp(-0.5 * warped_r / (len ** 2)) / (rcost) / len**2 # If we dont divide by len**2 here, case 0 can convert in Branin function

        # # res = amp * np.exp(-0.5 * warped_r) / np.sqrt(rcost)
        # # res = amp * np.exp(-0.5 * warped_r)

        # res = res / np.sqrt(rcost*len)
        # res = res / np.sqrt(rcost) / len
        # res = res / np.sqrt(rcost) / len * (np.sqrt(rr))
        # res = res/(np.sqrt(rcost)) ** np.sqrt(rr)
        # res = res / np.sqrt(rcost) * np.sqrt(rr)
        # res = np.add(rcost, 1e-8 * np.ones(rcost.shape))
        # res = amp * np.exp(-0.5 * warped_r)

        return res

    # def _radial_basis_cost_kernel(x, xprime, costx, costxprime, len, amp, costgp, alpha):
    #     # alpha = (totalBud - usedBud) / (totalBud - initBud)
    #     alpha = 1.0
    #     # alpha = 0.01
    #
    #     in_x = np.array(deepcopy(x))
    #     in_xprime = np.array(deepcopy(xprime))
    #
    #     #### Current kernels following ref ####
    #     noise_y = 0.1
    #     rcost = []
    #     rr = []
    #     # Kronecker = []
    #     for idx1 in range(np.shape(in_x)[0]):
    #         row = []
    #         rr_row = []
    #         # r1 = costx[idx1]
    #         r1 = costx[idx1]
    #         # r1 += 1.0
    #         r1 = r1 / (np.max(costgp.Ym)*2.0) + 1.0
    #         # r1 = r1*len
    #         # r1 = np.log(r1)
    #         # print("r1:", r1)
    #         r1 = r1
    #         # print("r1:",r1)
    #         # r1 = r1/np.sum(costgp.Ym)
    #
    #         for idx2 in range(np.shape(in_xprime)[0]):
    #             # r2 = costxprime[idx2]
    #             r2 = costxprime[idx2]
    #             # r2 += 1.0
    #             # r2 = r2*len
    #             r2 = r2 / (np.max(costgp.Ym)*2.0) + 1.0
    #             # r2 = np.log(r2)
    #             # print("r2:", r2)
    #             r2 = r2
    #             # r2 = r2/np.sum(costgp.Ym)
    #             # r1_r2 = ((np.abs(r1+r2) ** alpha))**2
    #             # r1_r2 = ((np.abs(r1 + r2)) ** 2)**alpha
    #             r1_r2 = ((np.abs(r1 + r2) ** alpha))*len
    #             r1_r2 = np.asscalar(r1_r2)
    #             # r1_r2 = np.log(r1_r2)
    #             # print("r1_r2:", r1_r2)
    #             # print("r1_r2:",r1_r2)
    #             # if r1_r2 > 0:
    #             #     r1_r2 *= len
    #             #     r1_r2 = r1_r2**2
    #             # else:
    #             #     r1_r2 = len**2
    #             row.append(r1_r2)
    #             rr_row.append(2*r1*r2)
    #             # if (np.linalg.norm(in_x[idx1] - in_xprime[idx2]) == 0):
    #             #     Kron_row.append(1.0*noise_y)
    #             # else:
    #             #     Kron_row.append(0.0)
    #         rcost.append(row)
    #         rr.append(rr_row)
    #         # Kronecker.append(Kron_row)
    #     rcost = np.array(rcost)
    #     # rr = np.array(rr)
    #     # rcost = costgp.Y_scaler.transform(rcost)
    #     rcost = np.add(rcost, 1e-8 * np.ones(rcost.shape))
    #     # rr = np.add(rr, 1e-8 * np.ones(rr.shape))
    #     r = euclidean_distances(in_x, in_xprime) ** 2
    #
    #     # warped_r = np.multiply(r, rcost)
    #     # warped_r = np.add(r,rcost) # Good
    #     warped_r = np.divide(r, rcost)
    #     # np.set_printoptions(precision=3)
    #     # print("r:", r)
    #     # print("rcost:", rcost)
    #     # print("warped_r:", warped_r)
    #     # exit()
    #     # amp = amp/rcost
    #     # res = amp * np.exp(-0.5 * warped_r/(len**2))
    #     res = amp * np.exp(-0.5 * warped_r)
    #     # res = res / np.sqrt(rcost*len)
    #     # res = res / np.sqrt(rcost)
    #     # res = res/(np.sqrt(rcost)) ** np.sqrt(rr)
    #     res = res / np.sqrt(rcost) * np.sqrt(rr)
    #     # res = np.add(rcost, 1e-8 * np.ones(rcost.shape))
    #     # res = amp * np.exp(-0.5 * warped_r)
    #
    #     return res

    @staticmethod
    def _radial_basis_cost_kernel_opt(x, xprime, costx, costxprime, len, amp, costgp, alpha):
        # alpha = (totalBud - usedBud) / (totalBud - initBud)
        # alpha = 1.0
        # alpha = 0.01

        in_x = np.array(deepcopy(x))
        in_xprime = np.array(deepcopy(xprime))

        #### Current kernels following ref ####
        noise_y = 0.1
        rcost = []
        # rr = []
        # Kronecker = []
        for idx1 in range(np.shape(in_x)[0]):
            row = []
            # rr_row = []
            # r1 = costx[idx1]
            r1 = costx[idx1]
            # r1 += 1.0
            r1 = r1/np.max(costgp.Ym) + 1.0
            # r1 = r1**alpha
            # r1 = r1*len
            r1 = np.log(r1)
            # r1 = r1
            # r1 = r1/np.sum(costgp.Ym)

            for idx2 in range(np.shape(in_xprime)[0]):
                # r2 = costxprime[idx2]
                r2 = costxprime[idx2]
                # r2 += 1.0
                # r2 = r2*len
                r2 = r2/np.max(costgp.Ym) + 1.0
                # r2 = r2**alpha
                r2 = np.log(r2)
                # r2 = r2
                # r2 = r2/np.sum(costgp.Ym)
                # r1_r2 = ((np.abs(r1+r2) ** alpha))**2
                # r1_r2 = ((np.abs(r1 + r2)) ** 2)**alpha
                # r1_r2 = ((np.abs(r1**2 + r2**2) ** alpha))
                r1_r2 = (r1**2 + r2**2) ** alpha
                r1_r2 = np.asscalar(r1_r2)
                # print("r1_r2:",r1_r2)
                # if r1_r2 > 0:
                #     r1_r2 *= len
                #     r1_r2 = r1_r2**2
                # else:
                #     r1_r2 = len**2
                row.append(r1_r2)
                # rr_row.append(r1*r2)
                # if (np.linalg.norm(in_x[idx1] - in_xprime[idx2]) == 0):
                #     Kron_row.append(1.0*noise_y)
                # else:
                #     Kron_row.append(0.0)
            rcost.append(row)
            # rr.append(rr_row)
            # Kronecker.append(Kron_row)
        rcost = np.array(rcost)
        # rr = np.array(rr)
        # rcost = costgp.Y_scaler.transform(rcost)
        rcost = np.add(rcost, 1e-8 * np.ones(rcost.shape))
        # rr = np.add(rr, 1e-8 * np.ones(rr.shape))
        r = euclidean_distances(in_x, in_xprime) ** 2

        # warped_r = np.multiply(r, rcost)
        # warped_r = np.add(r,rcost) # Good
        warped_r = np.divide(r, rcost)
        # np.set_printoptions(precision=3)
        # print("r:", r)
        # print("rcost:", rcost)
        # print("warped_r:", warped_r)
        # exit()
        # amp = amp/rcost
        res = amp * np.exp(-0.5 * warped_r/(len**2))
        # res = amp * np.exp(-0.5 * warped_r)
        # res = res / np.sqrt(rcost*len)
        res = res / np.sqrt(rcost) / len
        # res = res/(np.sqrt(rcost)) ** np.sqrt(rr)
        # res = res / np.sqrt(rcost) * np.sqrt(rr)
        # res = np.add(rcost, 1e-8 * np.ones(rcost.shape))
        # res = amp * np.exp(-0.5 * warped_r)

        return res

    @staticmethod
    def _mixed_basis_kernel(x, xprime, lengthScale, amp):
        # result = np.exp(-0.5 * np.square(euclidean_distances(x, xprime)))
        # result = np.exp(-0.5 * scipy.spatial.distance.cdist(x, xprime, 'euclidean'))
        # print("shape x:",np.array(x).shape, " shape xprime:", np.array(xprime).shape)
        #print("x: ", x)
        #print("xprime: ", xprime)
        # tmpX1, tmpX2 = np.array(x).T
        # tmpX1prime, tmpX2prime = np.array(xprime).T
        #
        # tmpX1 = tmpX1.reshape(len(tmpX1),1).tolist()
        # tmpX1prime = tmpX1prime.reshape(len(tmpX1prime),1).tolist()
        #
        # tmpX2 = tmpX2.reshape(len(tmpX2), 1).tolist()
        # tmpX2prime = tmpX2prime.reshape(len(tmpX2prime), 1).tolist()
        #print("tmpX1prime: ", tmpX1prime)
        #print("tmpX1:", tmpX1, " tmpX2:", tmpX2)
        #result = np.exp(-0.5 * euclidean_distances(x, xprime))
        #result = np.exp(-0.5 * euclidean_distances(tmpX1, tmpX1prime)) + np.exp(-0.5 * euclidean_distances(tmpX2, tmpX2prime))
        # result = np.exp(-0.5 * (euclidean_distances(tmpX1, tmpX1prime) + manhattan_distances(tmpX2, tmpX2prime)))
        x = np.array(x)
        xprime = np.array(xprime)
        p = 3
        sqdist = euclidean_distances(x, xprime)
        K = amp * np.exp(-2 * np.sin(np.pi * sqdist / p) ** 2 / (lengthScale ** 2) )
        return K

