from SceneSegmenter import SceneSegmenter


if __name__ == "__main__":

    segmenter = SceneSegmenter()
    segmenter.run(filename="./Falling.txt", 
                  ground_truth=[40, 70, 103, 134, 169, 197, 261, 308, 337, 362, 405], plot=True)