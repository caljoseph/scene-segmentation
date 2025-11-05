import os
from scene_segmenter import SceneSegmenter


def process_directory(directory_path, model_name="paraphrase-multilingual-MiniLM-L12-v2", **kwargs):
    segmenter = SceneSegmenter(model_name)
    total_tokens = 0
    tolerance = kwargs.get('tolerance', 3)
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
            _, identified_scenes, ground_truth_scenes, _, accuracies = segmenter.run(
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

    # OPTION 4: No classifier (just use peaks/valleys in embedding space)
    CLASSIFIER_PATH = None
    MODEL_NAME = "BAAI/bge-m3"

    # Set your data directory
    directory_path = "data/datasets/german_scene/stss_test_1"
    # =======================================================

    weighted_accuracies = process_directory(
        directory_path,
        model_name=MODEL_NAME,
        split_method="sentences",
        smooth="gaussian1d",
        sigma=1,
        diff="cosine",
        plot=True,
        print_accuracies=False,
        classifier_path=CLASSIFIER_PATH,
        tolerance=3
    )

    print("\nWeighted accuracies across all stories:")
    for metric, value in weighted_accuracies.items():
        print(f"  {metric}: {value:.4f}")