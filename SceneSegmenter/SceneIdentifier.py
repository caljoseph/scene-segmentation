import numpy as np
import re
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from BotFront import Bot


class SceneIdentifier():
    def __init__(self, classifier_path=None, llm_name=None):
        self.load_model(classifier_path)
        self.load_bot(llm_name)
        self.all_potential_scenes = []
        self.classifier_selected_scenes = []
        

    def identify(self, embeddings, sigma, smooth, diff, sentences,
                 embedder, classifier_path=None, 
                 alignment='center', k=4, target="minima", llm_name=None):
        
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
        maxima_indices = argrelextrema(deltaY_smoothed, np.greater)[0]

        if target == "minima":
            target_indices = minima_indices
        elif target == "maxima":
            target_indices = maxima_indices
        elif target == "both":
            target_indices = np.concatenate((minima_indices, maxima_indices))
        elif target == "random":
            target_indices = np.random.choice(np.arange(0, len(embeddings) + 1), len(embeddings)//15, replace=False)
        elif target.isdigit():
            target = int(target)
            target_indices = np.arange(0, len(embeddings), target)

        self.all_potential_scenes = target_indices.tolist()
        # APPLY CLASSIFIER
        if classifier_path:
            if self.classifier_path != classifier_path:
                #Reload if new classifier
                self.load_model(classifier_path)
            
            #Apply classifier
            target_indices = self.apply_classifier(target_indices, sentences, embedder, 
                                                   alignment=alignment, k=k)
        else:
            #If no classifier, just use the minima indicies 
            target_indices = target_indices.tolist()

        self.classifier_selected_scenes = target_indices

        # APPLY LLM
        if llm_name:
            if self.llm_name != llm_name:
                self.load_bot(llm_name)
            target_indices = self.apply_llm(target_indices, sentences, alignment, k)


        return deltaY, deltaY_smoothed, target_indices
     
    def apply_classifier(self, target_indices, sentences, embedder, alignment="center", k=2):
        import torch

        potential_scenes = []
        
        for index in target_indices:
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

        scenes = [target_indices[i] for i in range(len(classifications)) if classifications[i]]

        return scenes
    
    def apply_llm(self, target_indices, sentences, alignment="center", k=2):
        scene_windows = []

        for index in target_indices:
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
            
            scene_windows.append((sentences[start_idx:end_idx], start_idx)) #store sentence windows paired with ID's

        target_indices = []
        for window_pack in scene_windows:
            window, window_start_index = window_pack
            print("checking scene")
            scene_id, scene_found = self.bot.check_scene(window)
            print(f'ID: {scene_id}, found: {scene_found}')
            if scene_found:
                target_indices.append(scene_id + window_start_index)
                print(f"scene found at {target_indices[-1]}")

        print(f'Bot calls: {self.bot.API_calls}, \
              input tokens: {self.bot.prompt_tokens_total}, \
              generated tokens: {self.bot.response_tokens_total}')

        return target_indices



    def load_model(self, classifier_path):
        self.classifier_path = classifier_path
        if classifier_path:
            import torch #only import torch if its needed

            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"Using device: {device}")

            checkpoint = torch.load(classifier_path, map_location=device)
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

    def load_bot(self, llm_name):
        print(f'Loading bot using: {llm_name}')
        self.llm_name = llm_name
        if llm_name:
            self.bot = Bot(llm_name)
        else:
            if getattr(self, 'bot', None) is None: #no need to unload bot and lose token info
                self.bot = None
                
        