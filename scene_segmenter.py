import matplotlib.pyplot as plt
import re
import os
from sklearn.metrics import f1_score, precision_score, recall_score

from input_reader import InputReader
from embedder import Embedder
from scene_identifier import SceneIdentifier


#TODO - Additional accuracy measures
#TODO - provide option for just saving plots, instead of displaying them first

class SceneSegmenter():
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2',
                 classifier_path=None,
                 use_cache=True,
                 cache_dir="data/cache"):

        self.input_reader = InputReader()
        self.embedder = Embedder(model_name, use_cache=use_cache, cache_dir=cache_dir)
        self.scene_identifier = SceneIdentifier(classifier_path=classifier_path)

    #Input: filename for a txt file
    def run(self, filename, sigma=3, plot=False, print_accuracies=False, ground_truth_sentences=None,
            split_method="sentences",
            split_len=50, smooth="gaussian1d", diff="2norm", target="minima",
            classifier_path=None, tolerance=3):

        #check inputs
        if ".csv" not in filename and ".txt" not in filename:
            raise ValueError(f'{filename} is not a .txt or .csv file')
        if split_method not in ["sentences", "tokens_exact", "tokens_min"] and "token" not in split_method:
            raise ValueError(
                f'{split_method} is an invalid value for split_method, allowed values are: {["sentences", "tokens_exact", "tokens_min"]}')
        if not bool(re.fullmatch(r"(inf|\d+)norm", diff)) and "cos" not in diff:
            raise ValueError(f"{diff} is invalid difference measure. Valid values are ['pnorm', 'cosine']")
        if smooth not in ["gaussian1d", None]:
            raise ValueError(f"{smooth} is invalid smoothing method. Valid values are ['gaussian1d', None]")
        if classifier_path not in [None] and os.path.isfile('classifier_path'):
            raise ValueError(f"{classifier_path} not found. Use [None] or a valid classifier path.")
        if target not in ["minima", "min", "maxima", "max", "both", "random"] and not target.isdigit():
            raise ValueError(f"{target} is invalid target method. Must be 'min', 'max', 'both', 'random' or an integer")

        #Ground truth can be drawn from a CSV or input as an array corresponding to sentence number for scenes
        if ".csv" in filename:
            sentences, ground_truth_sentences, num_tokens = self.input_reader.read(filename, split_method, split_len)
        else:
            sentences, _, num_tokens = self.input_reader.read(filename, split_method, split_len)

        #Pass sentences through sentence embedders
        embeddings = self.embedder.generate_embeddings(sentences)

        #Initial identification of potential scenes
        deltaY, deltaY_smoothed, scenes = self.scene_identifier.identify(embeddings, sigma, smooth, diff, target=target,
                                                                         sentences=sentences, embedder=self.embedder,
                                                                         classifier_path=classifier_path,
                                                                         alignment='center', tolerance=tolerance)

        #Construct indentified and ground truth position arrays
        identified_scenes_tokens = self.convert_to_tokens(scenes, sentences)
        if ground_truth_sentences is not None:
            ground_truth_tokens = self.convert_to_tokens(ground_truth_sentences, sentences)
        else:
            ground_truth_tokens = None

        # Ensure both arrays start with 0 and remove the last element
        identified_scenes_tokens = self.adjust_token_array(identified_scenes_tokens)
        ground_truth_tokens = self.adjust_token_array(
            ground_truth_tokens) if ground_truth_tokens is not None else None

        accuracies = {}  # used for returning collected accuracy measures
        if ground_truth_sentences is not None and len(scenes) > 0:
            num_sentences = len(sentences)

            # Calculate pointwise dissimilarity
            pointwise_dissimilarity = self.calc_accuracy(scenes, ground_truth_sentences, num_sentences)

            # Calculate relaxed metrics (Zehe et al.)
            relaxed_precision, relaxed_recall, relaxed_f1 = self.calc_relaxed_metrics(
                scenes, ground_truth_sentences, num_sentences, tolerance=tolerance
            )

            # Optional: Calculate exact metrics for comparison
            exact_precision, exact_recall, exact_f1 = self.calc_exact_metrics(
                scenes, ground_truth_sentences, num_sentences
            )

            accuracies["Pointwise dissimilarity"] = pointwise_dissimilarity
            accuracies["Relaxed Precision (t={})".format(tolerance)] = relaxed_precision
            accuracies["Relaxed Recall (t={})".format(tolerance)] = relaxed_recall
            accuracies["Relaxed F1 (t={})".format(tolerance)] = relaxed_f1
            accuracies["Exact Precision"] = exact_precision
            accuracies["Exact Recall"] = exact_recall
            accuracies["Exact F1"] = exact_f1
            # for k in [0,1,2,4]:
            #     k_accuracy = self.count_scenes_within_k(ground_truth_sentences, scenes, k) / len(scenes) * 100
            #     accuracies[f"{k} ground truth accuracy"] = k_accuracy
            #     #print(f'{k_accuracy * 100}% of identified scenes within {k} of GT scenes')
            # for k in [0,1,2,4]:
            #     k_accuracy = self.count_scenes_within_k(scenes, ground_truth_sentences, k) / len(ground_truth_sentences) * 100
            #     accuracies[f"{k} scenes found"] = k_accuracy
            #     #print(f'{k_accuracy * 100}% of GT scenes within {k} of identified scenes')
            #
            if print_accuracies:
                for key in accuracies.keys():
                    print(f'{key}: {accuracies[key]}')
                print("")

        if plot:
            self.plot_scenes(deltaY_smoothed, scenes, sigma, filename, diff, ground_truth_sentences,
                             split_len=split_len, split_method=split_method)

        return deltaY_smoothed, identified_scenes_tokens, ground_truth_tokens, scenes, accuracies

    def calc_relaxed_metrics(self, predicted_borders, gold_borders, num_sentences, tolerance=3):
        """
        Calculate relaxed precision, recall, and F1-score as described in Zehe et al.

        Args:
            predicted_borders: List of predicted scene boundary sentence indices
            gold_borders: List of gold annotated scene boundary sentence indices
            num_sentences: Total number of sentences in the text
            tolerance: Maximum allowed distance (in sentences) for a match (default: 3)

        Returns:
            Tuple of (precision, recall, f1) for the BORDER class after applying tolerance
        """
        # Apply tolerance: move predicted borders to gold borders if within tolerance
        adjusted_predicted = []
        for pred in predicted_borders:
            # Find closest gold border
            closest_gold = None
            min_distance = float('inf')
            for gold in gold_borders:
                distance = abs(pred - gold)
                if distance < min_distance:
                    min_distance = distance
                    closest_gold = gold

            # If within tolerance, move to gold position; otherwise keep original
            if min_distance <= tolerance:
                adjusted_predicted.append(closest_gold)
            else:
                adjusted_predicted.append(pred)

        # Remove duplicates (multiple predictions might map to same gold border)
        adjusted_predicted = list(set(adjusted_predicted))

        # Create sentence-level binary labels (BORDER vs NO-BORDER)
        y_true = [1 if i in gold_borders else 0 for i in range(num_sentences)]
        y_pred = [1 if i in adjusted_predicted else 0 for i in range(num_sentences)]

        # Calculate metrics for the minority class (BORDER)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        return precision, recall, f1

    def calc_exact_metrics(self, predicted_borders, gold_borders, num_sentences):
        """
        Calculate exact precision, recall, and F1-score (sentence-level classification without tolerance).

        Args:
            predicted_borders: List of predicted scene boundary sentence indices
            gold_borders: List of gold annotated scene boundary sentence indices
            num_sentences: Total number of sentences in the text

        Returns:
            Tuple of (precision, recall, f1) for the BORDER class
        """
        # Create sentence-level binary labels (BORDER vs NO-BORDER)
        y_true = [1 if i in gold_borders else 0 for i in range(num_sentences)]
        y_pred = [1 if i in predicted_borders else 0 for i in range(num_sentences)]

        # Calculate metrics for the minority class (BORDER)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        return precision, recall, f1

    def convert_to_tokens(self, scene_indices, text_units):
        token_counts = []
        current_count = 0
        for i, unit in enumerate(text_units):
            if i in scene_indices:
                token_counts.append(current_count)
            current_count += len(unit.split())
        token_counts.append(current_count)  # Add total token count at the end
        return token_counts

    def adjust_token_array(self, token_array):
        if token_array is None:
            return None
        if token_array[0] != 0:
            token_array = [0] + token_array
        return token_array[:-1]  # Remove the last element (total token count)

    def plot_scenes(self, deltaY_smoothed, system_output, sigma, filename, diff="none", ground_truth=None,
                    split_len=None, split_method="sentences"):
        plt_1 = plt.figure(figsize=(25, 2))

        # Extract just the base filename without the path
        base_filename = os.path.basename(filename)

        if split_method == "sentences":
            plt.title(
                f"Difference in Sentence Embeddings for '{base_filename}' (sigma={sigma}, diff={diff}, {split_method}) using {self.embedder.model_name}")
        elif "token" in split_method:
            plt.title(
                f"Difference in Sentence Embeddings for '{base_filename}' (sigma={sigma}, diff={diff}, {split_method}, num tokens: {split_len}) using {self.embedder.model_name}")

        plt.xlabel("Sentence (narrative order)")
        plt.ylabel("Change in Embeddings")
        plt.xlim([0, len(deltaY_smoothed) - 1])
        plt.plot(deltaY_smoothed)
        if ground_truth is not None:
            for scene in ground_truth:
                plt.axvline(x=scene, color='r')
        for local_min in system_output:
            plt.plot(local_min, deltaY_smoothed[local_min], marker="o", markersize=5, markeredgecolor="black",
                     markerfacecolor="red")

        # Create 'Plots' directory if it doesn't exist
        os.makedirs('Plots', exist_ok=True)

        # Generate the save path using just the base filename
        save_path = os.path.join('Plots', f"{os.path.splitext(base_filename)[0]}_line_only_plot.png")

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(plt_1)  # Close the figure to free up memory

        print(f"Plot saved to: {save_path}")

    def calc_accuracy(self, scene_partitions1, scene_partitions2, max_tokens):
        # find which has the most tokens and assign the appropriate variable names
        large = []
        small = []
        if len(scene_partitions1) > len(scene_partitions2):
            large = scene_partitions1.copy()
            small = scene_partitions2.copy()
        else:
            large = scene_partitions2.copy()
            small = scene_partitions1.copy()

        # append a point a the beginning at 0 and end at max_tokens
        large.insert(0, 0)
        large.append(max_tokens)
        small.insert(0, 0)
        small.append(max_tokens)

        # calculate error of smaller set to larger set
        pointwise_error = 0
        for l in large:
            smallest = max_tokens
            for s in small:
                error = abs(l - s)
                if error < smallest:
                    smallest = error
            pointwise_error = pointwise_error + smallest
        # print(pointwise_error)

        return pointwise_error / max_tokens

    #A measure combining nearest scenes and number of scenes.
    def calc_accuracy_alt(self, system_output, ground_truth):
        #smallest distance from gt scenes to nearest system output scenes
        distances = []
        for scene_gt in ground_truth:
            nearest_distance = min([abs(scene_gt - scene_sys) for scene_sys in system_output])
            distances.append(nearest_distance)

        #number of scenes - difference in number of scenes divided by average number of scenes
        len_ratio = abs(len(system_output) - len(ground_truth)) / ((len(system_output) + len(ground_truth)) / 2)

        return sum(distances) / len(distances) * len_ratio

    def count_scenes_within_k(self, partition_1, partition_2, k):
        count = 0

        # Iterate over each scene in partition_2
        for scene_2 in partition_2:
            # Check if any scene in partition_1 is within distance k of scene_2
            within_k = any(abs(scene_2 - scene_1) <= k for scene_1 in partition_1)
            if within_k:
                count += 1

        return count
