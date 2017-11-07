from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal as mvn

class AudioCluster:

    def __init__(self, SegIdx2d, completeFeature):
        self.SegIdx2d = SegIdx2d
        self.completeFeature = completeFeature


    def __segLL__(self, feature_seg):
        LL_seg = np.sum(mvn.logpdf(feature_seg.T, mean = feature_seg.mean(axis = 1), cov = np.cov(feature_seg), allow_singular=True))
        return LL_seg

    def __getAllSegLL__(self, completeFeature):
        LL_AllSeg = []
        for elemTuple in self.SegIdx2d:
            tmpFeatureSeg = completeFeature[:, xrange(elemTuple[0], elemTuple[1])].copy()
            LL_AllSeg.append(self.__segLL__(tmpFeatureSeg))
        return np.array(LL_AllSeg)

    def __getDist__(self, SegIdx1, SegIdx2, LL_AllSeg, completeFeature):
        FeatureSeg1 = completeFeature[:, xrange(self.SegIdx2d[SegIdx1][0], self.SegIdx2d[SegIdx1][1])].copy()
        FeatureSeg2 = completeFeature[:, xrange(self.SegIdx2d[SegIdx2][0], self.SegIdx2d[SegIdx2][1])].copy()

        FeatureSegComb = np.hstack((FeatureSeg1, FeatureSeg2))
        LL0 = self.__segLL__(FeatureSegComb)

        return -LL0 + LL_AllSeg[SegIdx1] + LL_AllSeg[SegIdx2]

    def getPairwiseDist(self):
        LL_AllSeg = self.__getAllSegLL__(self.completeFeature)
        distMat = np.zeros(shape = (self.SegIdx2d.shape[0], self.SegIdx2d.shape[0]), dtype = float)
        for i in xrange(self.SegIdx2d.shape[0]):
            for j in xrange(i, self.SegIdx2d.shape[0]):
                if i == j:
                    distMat[i][j] = 0
                else:
                    distMat[i][j] = self.__getDist__(i, j, LL_AllSeg, self.completeFeature)
        return distMat + distMat.T

    # def Gc(self, clusterID):












