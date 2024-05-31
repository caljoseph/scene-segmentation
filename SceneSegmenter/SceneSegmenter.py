import matplotlib.pyplot as plt
import re



from InputReader import InputReader
from Embedder import Embedder
from SceneIdentifier import SceneIdentifier


#TODO - Additional accuracy measures
#TODO - provide option for just saving plots, instead of displaying them first

class SceneSegmenter():
    def __init__(self, model_name='all-MiniLM-L6-v2'):

        self.input_reader = InputReader()
        self.embedder = Embedder(model_name=model_name)
        self.scene_identifier = SceneIdentifier()

        


    #Input: filename for a txt file
    def run(self, filename, sigma=3, plot=False, ground_truth=None, split_method="sentences", 
            split_len=50, smooth="gaussian1d", diff="2norm", model_name='all-MiniLM-L6-v2',
            classifier=None):
        
        #model_name is checked inside the embedder 
        if self.embedder.model_name != model_name:
            print(f"Changing model from {self.embedder.model_name} to {model_name}")
            self.embedder.set_model(model_name)

        #check inputs
        if ".csv" not in filename and ".txt" not in filename:
            raise ValueError(f'{filename} is not a .txt or .csv file')
        if split_method not in ["sentences", "tokens_exact", "tokens_min"] and "token" not in split_method:
            raise ValueError(f'{split_method} is an invalid value for split_method, allowed values are: {["sentences", "tokens_exact", "tokens_min"]}')
        if not bool(re.fullmatch(r"(inf|\d+)norm", diff)) and "cos" not in diff:
            raise ValueError(f"{diff} is invalid difference measure. Valid values are ['pnorm', 'cosine']")
        if smooth not in ["gaussian1d", None]:
            raise ValueError(f"{smooth} is invalid smoothing method. Valid values are ['gaussian1d', None]")
        if classifier not in ["default", None]:
            raise ValueError(f"{smooth} is invalid smoothing method. Valid values are ['default']")

        #Ground truth can be drawn from a CSV or input as an array corresponding to sentence number for scenes
        if ".csv" in filename:
            split_sent, ground_truth, num_tokens = self.input_reader.read(filename, split_method, split_len)
        else:
            split_sent, _, num_tokens = self.input_reader.read(filename, split_method, split_len)

        embeddings = self.embedder.generate_embeddings(split_sent)

        deltaY, deltaY_smoothed, minima_indices = self.scene_identifier.identify(embeddings, sigma, smooth, diff)

        if classifier == "default":
            scenes = self.scene_identifier.apply_classifier(minima_indices, split_sent, self.embedder, classifier, 
                                                   alignment="center", k = 1)


        if ground_truth is not None:
            accuracy = self.calc_accuracy(minima_indices.tolist(), ground_truth, num_tokens)
            alt_accuracy = self.calc_accuracy_alt(minima_indices.tolist(), ground_truth)
            print(f'Accuracy: {accuracy}, Alt: {alt_accuracy}')

        if plot:
            self.plot_scenes(deltaY_smoothed, minima_indices.tolist(), sigma, filename, diff, ground_truth, split_len=split_len, split_method=split_method)

        return deltaY_smoothed, minima_indices.tolist()


    def plot_scenes(self, deltaY_smoothed, system_output, sigma, filename, diff="none" ,ground_truth=None, split_len=None, split_method="sentences"):
        plt_1 = plt.figure(figsize=(25, 2))
        if split_method == "sentences":
            plt.title(f"Difference in Sentence Embeddings for '{filename}' (sigma={sigma}, diff={diff}, {split_method}) using {self.embedder.model_name}")
        elif "token" in split_method:
            plt.title(f"Difference in Sentence Embeddings for '{filename}' (sigma={sigma}, diff={diff}, {split_method}, num tokens: {split_len}) using {self.embedder.model_name}")
            
        plt.xlabel("Sentence (narrative order)")
        plt.ylabel("Change in Embeddings")
        plt.xlim([0,len(deltaY_smoothed)-1])
        plt.plot(deltaY_smoothed)
        if ground_truth != None:
            for scene in ground_truth:
                plt.axvline(x = scene, color = 'r')
        for loclal_min in system_output:
            plt.plot(loclal_min, deltaY_smoothed[loclal_min], marker="o", markersize=5, markeredgecolor="black", markerfacecolor="red")
        plt.show()

    def calc_accuracy(self, scene_partitions1, scene_partitions2, max_tokens):
        accuracy = 0

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

        return pointwise_error/max_tokens

    #A measure combining nearest scenes and number of scenes.
    def calc_accuracy_alt(self, system_output, ground_truth):
        #smallest distance from gt scenes to nearest system output scenes
        distances = []
        for scene_gt in ground_truth:
            nearest_distance = min([abs(scene_gt - scene_sys) for scene_sys in system_output])
            distances.append(nearest_distance)

        #number of scenes - difference in number of scenes divided by average number of scenes
        len_ratio = abs(len(system_output) - len(ground_truth)) / ((len(system_output) + len(ground_truth))/2)

        return sum(distances)/len(distances) * len_ratio
