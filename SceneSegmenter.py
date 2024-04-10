import nltk
import nltk.data
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


class SceneSegmenter():
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        nltk.download('punkt')

        #models = ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'all-mpnet-base-v2']

        self.model = SentenceTransformer(model_name)


    #generates embeddings from a txt file
    def generate_embeddings(self, filename):
        text = open(filename, "r", encoding="utf8").read()
        tokens = nltk.word_tokenize(text)
        sent_text = nltk.sent_tokenize(text)

        embeddings = self.model.encode(sent_text)
        outputs = []
        for embedding in embeddings:
            outputs.append(embedding)
        return outputs, len(tokens)


    #Input: filename for a txt file
    def run(self, filename, sigma=3, plot=False, ground_truth=None):
        deltaY = []

        np_outputs, token_length = self.generate_embeddings(filename)

        for n in range(len(np_outputs)-1):
            deltaY.append(np.linalg.norm(np_outputs[n]-np_outputs[n+1]))

        deltaY_smoothed = gaussian_filter1d(deltaY, sigma)

        minima_indices = argrelextrema(deltaY_smoothed, np.less)[0]

        if plot:
            self.plot_scenes(deltaY_smoothed, minima_indices.tolist(), sigma, filename, ground_truth)

        if ground_truth is not None:
            accuracy = self.calc_accuracy(deltaY_smoothed, ground_truth, token_length)
            alt_accuracy = self.calc_accuracy_alt(deltaY_smoothed, ground_truth)
            print(f'Accuracy: {accuracy}, Alt: {alt_accuracy}')

        return deltaY_smoothed, minima_indices.tolist()


    def plot_scenes(self, deltaY_smoothed, system_output, sigma, filename, ground_truth=None):
        plt_1 = plt.figure(figsize=(25, 2))
        plt.title(f"Difference in Sentence Embeddings for '{filename}' (sigma={sigma})")
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
        #avg distance from gt scenes to nearest system output scenes
        distances = []
        for scene_gt in ground_truth:
            nearest_distance = min([abs(scene_gt - scene_sys) for scene_sys in system_output])
            distances.append(nearest_distance)

        #number of scenes - difference in number of scenes divided by average number of scenes
        len_ratio = abs(len(system_output) - len(ground_truth)) / ((len(system_output) + len(ground_truth))/2)
        rating = (len_ratio) *10

        return sum(distances)/len(distances) + rating

if __name__ == "__main__":

    segmenter = SceneSegmenter()
    segmenter.run(filename="./Data/Text Files/Observer_1-A_Warm_Home.txt", 
                  ground_truth=[20, 38, 136, 257, 369, 395, 427, 548, 623], plot=True)
