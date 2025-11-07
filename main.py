import os
from scene_segmenter import SceneSegmenter
from candidate_selectors import GaussianCandidateSelector, SavitzkyGolayCandidateSelector


def process_directory(directory_path, model_name="paraphrase-multilingual-MiniLM-L12-v2", **kwargs):
    segmenter = SceneSegmenter(model_name)
    total_tokens = 0
    tolerance = kwargs.get('tolerance', 3)

    # Process all CSV files in directory
    weighted_accuracies = {
        "Pointwise dissimilarity": 0,
        "Relaxed Precision (t={})".format(tolerance): 0,
        "Relaxed Recall (t={})".format(tolerance): 0,
        "Relaxed F1 (t={})".format(tolerance): 0,
        "Exact Precision": 0,
        "Exact Recall": 0,
        "Exact F1": 0
    }

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            _, identified_scenes, ground_truth_scenes, _, accuracies, _ = segmenter.run(
                filename=file_path,
                **kwargs
            )

            # Count ground truth tokens
            story_tokens = len(ground_truth_scenes)
            total_tokens += story_tokens

            # Accumulate weighted accuracies
            for metric, value in accuracies.items():
                weighted_accuracies[metric] += value * story_tokens

    # Calculate final weighted averages
    for metric in weighted_accuracies:
        weighted_accuracies[metric] /= total_tokens

    return weighted_accuracies


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # OPTION 1: Use an LLM as classifier (no embedding model needed, more expensive)
    # Supported LLM names: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, etc.
    # CLASSIFIER_PATH = "gpt-4o-mini"
    # MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Still used for initial embeddings

    # OPTION 2: Use a trained sequence classifier (LSTM/CNN - NEW!)
    # Make sure MODEL_NAME matches the embedder used to train the classifier!
    # CLASSIFIER_PATH = "data/models/lstm_BAAI-bge-m3_german-stss_512h.pth"
    # MODEL_NAME = "BAAI/bge-m3"  # Must match the classifier's embedding dimensions

    # OPTION 3: Use old-style concatenated classifier (backward compatibility)
    # CLASSIFIER_PATH = "classifiers/classifier_BAAI-bge-m3_1024_dims.pth"
    # MODEL_NAME = "BAAI/bge-m3"

    # OPTION 2: Use fine-tuned LLM classifier
    CLASSIFIER_PATH = "data/models/llm-3b_exp7_low_rank"
    MODEL_NAME = "BAAI/bge-m3"

    # Set your data directory
    directory_path = "data/datasets/german_scene/test_full"

    # Mode options:
    # - "optima": Use embedding peaks/valleys + classifier refinement (default)
    # - "sliding_window": Pure LLM - classify every 7-sentence window
    MODE = "optima"

    # Candidate selection strategy (for optima mode):
    # - "gaussian": Original Gaussian smoothing + local minima (sigma=0.8)
    # - "savitzky_golay": Savitzky-Golay polynomial smoothing (window=11, poly=3)
    CANDIDATE_STRATEGY = "savitzky_golay"
    # =======================================================

    # Create candidate selector based on strategy
    if CANDIDATE_STRATEGY == "gaussian":
        candidate_selector = GaussianCandidateSelector(sigma=0.8, diff_metric="2norm", target="minima")
    elif CANDIDATE_STRATEGY == "savitzky_golay":
        candidate_selector = SavitzkyGolayCandidateSelector(window_length=11, polyorder=3, distance_metric="cosine")
    else:
        raise ValueError(f"Invalid CANDIDATE_STRATEGY: {CANDIDATE_STRATEGY}. Must be 'gaussian' or 'savitzky_golay'")

    weighted_accuracies = process_directory(
        directory_path,
        model_name=MODEL_NAME,
        split_method="sentences",
        plot=True,
        print_accuracies=False,
        classifier_path=CLASSIFIER_PATH,
        tolerance=3,
        mode=MODE,
        candidate_selector=candidate_selector
    )

    print("\nWeighted accuracies across all stories:")
    for metric, value in weighted_accuracies.items():
        print(f"  {metric}: {value:.4f}")