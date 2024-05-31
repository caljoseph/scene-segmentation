import numpy as np
import re
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


class SceneIdentifier():
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

        #Apply classifier model here

        return deltaY, deltaY_smoothed, minima_indices
     
     def apply_classifier(self, minima_indices, sentences, embedder, classifier, alignment="center", k=1):
        # TODO - Add warning if the embedder is different from what the classifier used
        # if embedder != classifier.embedder:
        #     print("Warning: The embedder used is different from the one the classifier was trained with.")

        # TODO - 
        potential_scenes = []
        
        for index in minima_indices:
            if alignment == "center":
                start_idx = max(0, index - k)
                end_idx = min(len(sentences), index + k + 1)
            elif alignment == "left":
                start_idx = max(0, index)
                end_idx = min(len(sentences), index + 2 * k + 1)
            elif alignment == "right":
                start_idx = max(0, index - 2 * k)
                end_idx = min(len(sentences), index + 1)
            else:
                raise ValueError("Invalid alignment argument. Choose from 'left', 'center', or 'right'.")
            
            combined_scene = ' '.join(sentences[start_idx:end_idx])
            potential_scenes.append(combined_scene)
        
        # Process potential_scenes using the embedder and classifier
        embeddings = embedder.generate_embeddings(potential_scenes)
        classifications = classifier.predict(embeddings)

        #TODO - make sure that the classifications are in a meaningful format
        
        return classifications
                
        