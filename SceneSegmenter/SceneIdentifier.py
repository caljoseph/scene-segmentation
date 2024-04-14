import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


class SceneIdentifier():
    def __init__(self, diff = "2norm", smooth="gaussian1d"):
        self.difference_measure = diff
        if self.difference_measure not in ["2norm"]:
            raise ValueError(f"{self.difference_measure} is invalid difference measure. Valid values are ['2norm']")
        self.smooth = smooth
        if self.smooth not in ["gaussian1d", None]:
            raise ValueError(f"{self.smooth} is invalid smoothing method. Valid values are ['gaussian1d', None]")


    def identify(self, embeddings, sigma):
        deltaY = [] #differences between embeddings

        if self.difference_measure == "2norm":
            for n in range(len(embeddings)-1):
                deltaY.append(np.linalg.norm(embeddings[n]-embeddings[n+1]))


        if self.smooth == "gaussian1d":
            deltaY_smoothed = gaussian_filter1d(deltaY, sigma)
        elif self.smooth is None:
            deltaY_smoothed = deltaY

        minima_indices = argrelextrema(deltaY_smoothed, np.less)[0]

        return deltaY, deltaY_smoothed, minima_indices
    

    def get_info(self):
        return self.difference_measure, self.smooth
    

    def print_info(self):
        print(f"Scene Identifier Instance: \n\tDifference Measure - {self.difference_measure}\n\tSmoothing - {self.smooth}")