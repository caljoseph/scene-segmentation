from SceneSegmenter import SceneSegmenter
from GUI import SE_GUI


# '''GUI Approach, allows for input into the GUI for running the system'''
# if __name__ == "__main__":
#
#     segmenter = SceneSegmenter()
#
#     gui = SE_GUI(segmenter.run)


'''Code Based Approach, allows for inputs to be manually input into an object'''
if __name__ == "__main__":

    segmenter = SceneSegmenter()

    _, identified_scenes, ground_truth_scenes, _, _ = segmenter.run(
        filename="./Annotated_CSVs/The_Night_Wire.csv",
        split_method="sentences",
        smooth="gaussian1d",
        sigma=2.5,
        plot=True,
        print_accuracies=True,
        classifier_path="./Classifiers/classifier_3_layer.pth",
        llm_name=None,
        k=4
    )

    print("Identified scenes (in tokens):", identified_scenes)
    print("Ground truth scenes (in tokens):", ground_truth_scenes)
    
