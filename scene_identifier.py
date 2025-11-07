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
                 alignment='center', tolerance=3, mode="optima", window_size=7,
                 ground_truth=None, candidate_selector=None):

        # SLIDING WINDOW MODE: Classify every possible window
        if mode == "sliding_window":
            # Generate all possible window center positions
            target_indices = self._construct_sliding_windows(sentences, window_size=window_size)

            # Apply classifier (required for sliding window mode)
            if classifier_path:
                if self.classifier_path != classifier_path:
                    self.load_model(classifier_path)

                target_indices = self.apply_classifier(target_indices, sentences, embedder,
                                                       alignment=alignment, tolerance=tolerance)

                # Print LLM usage stats if using LLM classifier
                if self.classifier_type == 'llm':
                    stats = self.classifier.get_usage_stats()
                    print(f"LLM API calls: {stats['api_calls']}, tokens: {stats['total_tokens']} (prompt: {stats['prompt_tokens']}, completion: {stats['completion_tokens']})")
            else:
                # Sliding window requires a classifier
                print("Warning: sliding_window mode requires a classifier. No scenes detected.")
                target_indices = []

            self.all_potential_scenes = target_indices
            self.classifier_selected_scenes = target_indices

            # Compute embedding deltaY for plotting (even though not used for detection)
            deltaY, deltaY_smoothed = self._compute_embedding_delta(embeddings, smooth, diff, sigma)
            return deltaY, deltaY_smoothed, target_indices

        # OPTIMA MODE: Use embedding peaks/valleys
        # Compute embedding differences (for plotting)
        deltaY, deltaY_smoothed = self._compute_embedding_delta(embeddings, smooth, diff, sigma)

        # Use candidate selector (required)
        if candidate_selector is None:
            raise ValueError("candidate_selector is required for optima mode. Use GaussianCandidateSelector or SavitzkyGolayCandidateSelector.")

        target_indices = np.array(candidate_selector.select_candidates(embeddings))

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
            # Fine-tuned LLM: tokenize and classify in batches
            import torch
            combined_scenes = [' '.join(window) for window in potential_scenes]

            # Process in batches to avoid OOM (increased for H100)
            batch_size = 64
            all_predictions = []

            for i in range(0, len(combined_scenes), batch_size):
                batch = combined_scenes[i:i+batch_size]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Classify batch
                with torch.no_grad():
                    outputs = self.classifier(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())

            classifications = (np.array(all_predictions) == 1)

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

                # Load tokenizer first (matches training setup)
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))

                # Configure padding to match training
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                # For decoder models, pad on the left (matches training)
                tokenizer.padding_side = 'left'

                # Check if it's a LoRA adapter or full model
                if (model_path / 'adapter_config.json').exists():
                    # LoRA adapter - need to load base model + adapter
                    import json
                    with open(model_path / 'adapter_config.json', 'r') as f:
                        adapter_config = json.load(f)
                    base_model_name = adapter_config.get('base_model_name_or_path')

                    # Load base model with pad_token_id (matches training)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_name,
                        num_labels=2,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    # Load LoRA adapter
                    model = PeftModel.from_pretrained(model, str(model_path))
                    model = model.merge_and_unload()  # Merge LoRA weights for faster inference
                else:
                    # Full fine-tuned model
                    model = AutoModelForSequenceClassification.from_pretrained(
                        str(model_path),
                        num_labels=2,
                        pad_token_id=tokenizer.pad_token_id
                    )

                # Ensure model config has the pad token set (matches training)
                model.config.pad_token_id = tokenizer.pad_token_id

                model.to(device)
                model.eval()

                self.classifier = model
                self.tokenizer = tokenizer
                self.device = device

                print(f"✓ Loaded fine-tuned LLM classifier: {classifier_path} (device: {device})")

        else:
            self.classifier = None
            self.classifier_type = None

    def _construct_sliding_windows(self, sentences, window_size=7):
        """
        Generate all possible sliding windows of given size.

        Args:
            sentences: List of sentence strings
            window_size: Number of sentences per window (default 7)

        Returns:
            window_positions: List of center indices for each window
        """
        window_positions = []
        half_window = window_size // 2

        # Generate windows for all valid center positions
        for center in range(half_window, len(sentences) - half_window):
            window_positions.append(center)

        return window_positions

    def _compute_embedding_delta(self, embeddings, smooth, diff, sigma):
        """
        Compute the difference between consecutive embeddings for plotting.

        Args:
            embeddings: List of sentence embeddings
            smooth: Smoothing method ("gaussian1d" or None)
            diff: Difference metric (e.g., "2norm", "cosine")
            sigma: Gaussian smoothing parameter

        Returns:
            Tuple of (deltaY, deltaY_smoothed)
        """
        deltaY = []

        # Regular expression for extracting norms
        if bool(re.fullmatch(r"(inf|\d+)norm", diff)):
            if "inf" in diff:
                p = np.inf
            else:
                p = int(re.search(r"(\d+)norm", diff).group(1))
            for n in range(len(embeddings)-1):
                deltaY.append(np.linalg.norm(embeddings[n]-embeddings[n+1], p))

        # Cosine similarity
        if "cos" in diff:
            for n in range(len(embeddings)-1):
                norm_product = (np.linalg.norm(embeddings[n]) * np.linalg.norm(embeddings[n+1]))
                if norm_product != 0:
                    deltaY.append(np.dot(embeddings[n], embeddings[n+1]) / norm_product)
                else:
                    deltaY.append(0)

        # Apply smoothing
        if smooth == "gaussian1d":
            deltaY_smoothed = gaussian_filter1d(deltaY, sigma)
        elif smooth is None:
            deltaY_smoothed = np.array(deltaY)

        return deltaY, deltaY_smoothed

