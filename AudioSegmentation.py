from __future__ import division
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

class AudioSegmentation:
    def __init__(self, 
                 features,
                 VAD_removal = True):
        self.features = features.copy()
        self.VAD_removal = VAD_removal

    def __helperSeg__(self, 
                      features, 
                      startFrame = 0, 
                      endFrame = 200, 
                      incFrame = 10, 
                      mixture = False, 
                      Lambda = 1):

        analFrame = features[:, startFrame:endFrame]
        Nz = endFrame - startFrame
        if mixture:
            gmm = GaussianMixture(n_components = 2, random_state = 101).fit(analFrame.T)
            post_prob = gmm.predict_proba(analFrame.T)
            L0 = np.sum(np.log(post_prob.dot(gmm.weights_.reshape(2,1))))
        else:
            d = features.shape[0]
            L0 = np.sum(np.log(mvn.pdf(analFrame.T, mean = analFrame.mean(axis = 1), cov = np.cov(analFrame), allow_singular=True)))
            L0 += Lambda/2 * (d + d * (d + 1) / 2) * np.log(Nz)
        
        for Nx in range(startFrame + 2 * incFrame, endFrame - incFrame, incFrame):
            features_x = features[:, startFrame:Nx]
            features_y = features[:, Nx: endFrame]
            L1_x = np.sum(mvn.logpdf(features_x.T, mean = features_x.mean(axis = 1), cov = np.cov(features_x), allow_singular=True))
            L1_y = np.sum(mvn.logpdf(features_y.T, mean = features_y.mean(axis = 1), cov = np.cov(features_y), allow_singular=True))
            L1 = L1_x + L1_y
            
            if L1 - L0 >= 0:
                return Nx
        return -1
            
    def Segmentation(self,
                     min_wind = 200,
                     inc_wind = 50,
                     max_wind = 500,
                     mov_frame = 10,
                     mixture = False,
                     Lambda = 1):
        self.Seg = []
        currentST = 0
        currentET = min_wind
        currentSize = currentET - currentST
        # initialization
        turnPt = self.__helperSeg__(self.features, startFrame = currentST, endFrame = currentET, incFrame = mov_frame, mixture = mixture, Lambda = Lambda)
        if turnPt != -1:
            currentST = turnPt
            currentET = turnPt + min_wind
            currentSize = currentET - currentST
            self.Seg.append(turnPt)
            turnPt = -1
        while currentST < self.features.shape[1]:
            while currentSize <= max_wind and turnPt == -1:
    #             print(currentET)
                currentET += inc_wind
                if currentET > self.features.shape[1]:
                    currentET = self.features.shape[1]
                    turnPt = self.__helperSeg__(self.features, startFrame = currentST, endFrame = currentET, incFrame = mov_frame, mixture = mixture, Lambda = Lambda)
                    break
                currentSize = currentET - currentST
                turnPt = self.__helperSeg__(self.features, startFrame = currentST, endFrame = currentET, incFrame = mov_frame, mixture = mixture, Lambda = Lambda)
            
            if currentSize > max_wind:
                currentST = currentST + max_wind - min_wind
                currentET = currentST + min_wind
                currentSize = currentET - currentST
            elif turnPt != -1:
                self.Seg.append(turnPt)
                currentST = turnPt
                currentET = currentST + min_wind
                currentSize = currentET - currentST
                turnPt = -1
            else:
                return self.Seg
        return self.Seg

    def Visualizer(self, tmin, tmax, AudioFeatures, AudioActDet = None, xticks_gap = 5):
        plt.figure(figsize=(18,6))
        plt.title('Segmentation Result')
        im = plt.imshow(np.flipud(AudioFeatures.mfcc[:, np.where((AudioFeatures.time_ins >= tmin) & (AudioFeatures.time_ins <= tmax))[0]]), 
                            aspect = 'auto', 
                            extent = [tmin, tmax, 0, AudioFeatures.mfcc.shape[0]], 
                            interpolation = 'nearest')
        if self.VAD_removal:
            # self.Seg.extend(list(AudioActDet.silence_seg))
            segTime = AudioFeatures.time_ins[AudioActDet.idx_act[self.Seg]]
            deadTime = AudioFeatures.time_ins[AudioActDet.silence_seg]
        else:
            segTime = AudioFeatures.time_ins[self.Seg]
        segTimePlot = segTime[(segTime >= tmin)&(segTime <= tmax)]
        deadTimePlot = deadTime[(deadTime >= tmin)&(deadTime <= tmax)]
        plt.vlines(segTimePlot, 0, 12, linewidth = 1)
        plt.vlines(deadTimePlot, 0, 12, linewidth = 1, linestyles = '--', color = 'm')
        plt.xticks(np.arange(tmin, tmax, xticks_gap))
        plt.show()

        return segTime, deadTime

