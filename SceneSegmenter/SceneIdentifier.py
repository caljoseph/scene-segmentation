import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


class SceneIdentifier():
    def __init__(self, diff = "2norm", smooth="gaussian1d"):
        self.difference_measure = diff
        


    def identify(self, embeddings, sigma, smooth, diff):

        deltaY = [] #differences between embeddings

        if diff == "2norm":
            for n in range(len(embeddings)-1):
                deltaY.append(np.linalg.norm(embeddings[n]-embeddings[n+1]))


        if smooth == "gaussian1d":
            deltaY_smoothed = gaussian_filter1d(deltaY, sigma)
        elif smooth is None:
            deltaY_smoothed = deltaY

        minima_indices = argrelextrema(deltaY_smoothed, np.less)[0]

        return deltaY, deltaY_smoothed, minima_indices
    