import numpy as np
import re
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


class SceneIdentifier():
    def __init__(self, diff = "2norm", smooth="gaussian1d"):
        self.difference_measure = diff
        


    def identify(self, embeddings, sigma, smooth, diff):

        deltaY = [] #differences between embeddings

        # Regular expression for extracting norms
        if bool(re.fullmatch(r"(inf|\d+)norm", diff)):
            if "inf" in diff:
                p = np.inf
            else:
                p = int(re.search(r"(\d+)norm", diff).group(1))
            for n in range(len(embeddings)-1):
                deltaY.append(np.linalg.norm(embeddings[n]-embeddings[n+1], p))

        # Corrected typo in "cosine" and improved condition
        if "cos" in diff:
            for n in range(len(embeddings)-1):
                norm_product = (np.linalg.norm(embeddings[n]) * np.linalg.norm(embeddings[n+1]))
                if norm_product != 0:  # Avoid division by zero
                    deltaY.append(np.dot(embeddings[n], embeddings[n+1]) / norm_product)
                else:
                    deltaY.append(0)  # Append 0 or some other placeholder for zero norm vectors

        if smooth == "gaussian1d":
            deltaY_smoothed = gaussian_filter1d(deltaY, sigma)
        elif smooth is None:
            deltaY_smoothed = np.array(deltaY)

        

        minima_indices = argrelextrema(deltaY_smoothed, np.less)[0]

        return deltaY, deltaY_smoothed, minima_indices
    