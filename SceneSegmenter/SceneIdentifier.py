import numpy as np
import re
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


class SceneIdentifier():
    def __init__(self, classifier_path=None):
        self.load_model(classifier_path)

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
     
    def apply_classifier(self, minima_indices, sentences, embedder, alignment="center", k=1):
        import torch
        # TODO - Add warning if the embedder is different from what the classifier used
        # if embedder != classifier.embedder:
        #     print("Warning: The embedder used is different from the one the classifier was trained with.")

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
        embeddings = torch.tensor(embeddings)
        classifications = self.classifier(embeddings) > 0.0

        scenes = [minima_indices[i] for i in range(len(classifications)) if classifications[i]]

        return scenes
     

    def load_model(self, classifier_path):
        self.classifier_path = classifier_path
        if classifier_path:
            import torch #only import torch if its needed TODO - add this to requirements.txt? 
            checkpoint = torch.load(classifier_path)
            model_class = checkpoint['model_architecture']
            model = model_class()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set the model to evaluation mode

            embedder_name = checkpoint.get('embedder_name', None)  # Use .get to avoid KeyError
            if embedder_name:
                print(f"Classifier from {classifier_path} trained on embedder {embedder_name} now loaded")
            else:
                print(f"Classifier from {classifier_path} now loaded. Warning: classifier embedder not specified.")

            self.classifier = model
        
        else:
            self.classifier = None
            print(f'No classifier loaded in SceneIdentifier ')
                
        