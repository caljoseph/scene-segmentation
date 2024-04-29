from SceneSegmenter import SceneSegmenter
from GUI import SE_GUI


if __name__ == "__main__":

    segmenter = SceneSegmenter()

    gui = SE_GUI(segmenter.run)

    # segmenter.run(filename="./Falling.txt", 
    #               ground_truth=[40, 70, 103, 134, 169, 197, 261, 308, 337, 362, 405],
    #               split_method="sentences",
    #               smooth="gaussian1d",
    #               plot=True)
    
