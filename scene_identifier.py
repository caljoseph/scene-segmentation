import numpy as np
import re
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from bot_front import SceneLLM
from models import SimpleClassifier, XGBoostClassifier
from classifier_registry import get_classifier_type


class SceneIdentifier():
    def __init__(self, classifier_path=None):
        self.load_model(classifier_path)
        self.all_potential_scenes = []
        self.classifier_selected_scenes = []


    def identify(self, embeddings, sigma, smooth, diff, sentences,
                 embedder, classifier_path=None,
                 alignment='center', tolerance=3, target="minima"):

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
                                                   alignment=alignment, tolerance=tolerance)

            # Print LLM usage stats if using LLM classifier
            if self.classifier_type == 'llm':
                stats = self.classifier.get_usage_stats()
                print(f"LLM API calls: {stats['api_calls']}, tokens: {stats['total_tokens']} (prompt: {stats['prompt_tokens']}, completion: {stats['completion_tokens']})")
        else:
            #If no classifier, just use the minima indicies
            target_indices = target_indices.tolist()

        self.classifier_selected_scenes = target_indices

        return deltaY, deltaY_smoothed, target_indices

    def apply_classifier(self, target_indices, sentences, embedder, alignment="center", tolerance=3):
        potential_scenes = []

        for index in target_indices:
            if alignment == "center":
                start_idx = max(0, index - tolerance)
                end_idx = min(len(sentences), index + tolerance + 1)
            elif alignment == "left":
                start_idx = max(0, index)
                end_idx = min(len(sentences), index + 2 * tolerance + 1)
            elif alignment == "right":
                start_idx = max(0, index - 2 * tolerance)
                end_idx = min(len(sentences), index + 1)
            else:
                raise ValueError("Invalid alignment argument. Choose from 'left', 'center', or 'right'.")

            # Keep sentences as a list for sequence classifiers
            sentence_window = sentences[start_idx:end_idx]
            potential_scenes.append(sentence_window)

        # Handle different classifier types
        if self.classifier_type == 'pytorch':
            import torch

            # Check if it's a sequence classifier (LSTM/CNN)
            is_sequence_classifier = hasattr(self.classifier, 'lstm') or hasattr(self.classifier, 'convs')

            if is_sequence_classifier:
                # Sequence classifier: embed each sentence separately
                all_embeddings = []
                for window in potential_scenes:
                    # Embed each sentence in the window
                    window_embeddings = embedder.generate_embeddings(window)
                    all_embeddings.append(window_embeddings)

                # Stack into (batch, seq_len, embedding_dim)
                embeddings_array = np.array(all_embeddings)  # (num_windows, window_size, embedding_dim)
                embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32)
            else:
                # Old classifier: concatenate sentences and embed as one
                combined_scenes = [' '.join(window) for window in potential_scenes]
                embeddings = embedder.generate_embeddings(combined_scenes)
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

            # Classify
            classifications = self.classifier(embeddings_tensor) > 0.0

        elif self.classifier_type == 'xgboost':
            # XGBoost: concatenate and embed
            combined_scenes = [' '.join(window) for window in potential_scenes]
            embeddings = embedder.generate_embeddings(combined_scenes)
            predictions = self.classifier.predict(embeddings)
            classifications = predictions == 1

        elif self.classifier_type == 'llm':
            # LLM: works directly with text
            combined_scenes = [' '.join(window) for window in potential_scenes]
            print(f"Classifying {len(combined_scenes)} candidate windows with LLM sequentially...")
            classifications = self.classifier.classify_batch(combined_scenes, batch_size=1)
            print(f"  Done! Classified {len(combined_scenes)} windows.")
            self.classifier.print_sample_responses()

        elif self.classifier_type == 'llm-finetune':
            # Fine-tuned LLM: tokenize and classify
            import torch
            combined_scenes = [' '.join(window) for window in potential_scenes]

            # Tokenize
            inputs = self.tokenizer(
                combined_scenes,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Classify
            with torch.no_grad():
                outputs = self.classifier(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                classifications = (predictions == 1).cpu().numpy()

        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

        scenes = [target_indices[i] for i in range(len(classifications)) if classifications[i]]

        return scenes



    def load_model(self, classifier_path):
        self.classifier_path = classifier_path
        if classifier_path:
            # Determine classifier type using the registry
            classifier_type = get_classifier_type(classifier_path)
            self.classifier_type = classifier_type

            if classifier_type == 'llm':
                # Load LLM classifier
                self.classifier = SceneLLM(model=classifier_path)
                print(f"✓ Loaded LLM classifier: {classifier_path}")

            elif classifier_type == 'pytorch':
                # Load PyTorch model
                import torch
                from pathlib import Path

                device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

                checkpoint = torch.load(classifier_path, map_location=device)
                embedding_dim = checkpoint['embedding_dim']
                model = SimpleClassifier(embedding_dim)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                embedder_name = checkpoint.get('embedder_name', None)
                if embedder_name:
                    print(f"✓ Loaded PyTorch classifier: {classifier_path} (device: {device}, embedder: {embedder_name}, dims: {embedding_dim})")
                else:
                    print(f"✓ Loaded PyTorch classifier: {classifier_path} (device: {device}, dims: {embedding_dim}, warning: embedder not specified)")

                self.classifier = model

            elif classifier_type == 'xgboost':
                # Load XGBoost model
                model = XGBoostClassifier()
                model.load(str(classifier_path))
                print(f"✓ Loaded XGBoost classifier: {classifier_path}")
                self.classifier = model

            elif classifier_type == 'llm-finetune':
                # Load fine-tuned LLM classifier
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                from peft import PeftModel
                from pathlib import Path

                device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

                model_path = Path(classifier_path)

                # Check if it's a LoRA adapter or full model
                if (model_path / 'adapter_config.json').exists():
                    # LoRA adapter - need to load base model + adapter
                    import json
                    with open(model_path / 'adapter_config.json', 'r') as f:
                        adapter_config = json.load(f)
                    base_model_name = adapter_config.get('base_model_name_or_path')

                    # Load base model
                    model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_name,
                        num_labels=2
                    )
                    # Load LoRA adapter
                    model = PeftModel.from_pretrained(model, str(model_path))
                    model = model.merge_and_unload()  # Merge LoRA weights for faster inference
                else:
                    # Full fine-tuned model
                    model = AutoModelForSequenceClassification.from_pretrained(
                        str(model_path),
                        num_labels=2
                    )

                model.to(device)
                model.eval()

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                self.classifier = model
                self.tokenizer = tokenizer
                self.device = device

                print(f"✓ Loaded fine-tuned LLM classifier: {classifier_path} (device: {device})")

        else:
            self.classifier = None
            self.classifier_type = None

