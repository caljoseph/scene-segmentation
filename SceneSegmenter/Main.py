from SceneSegmenter import SceneSegmenter
from GUI import SE_GUI


'''GUI Approach, allows for input into the GUI for running the system'''
if __name__ == "__main__":

    segmenter = SceneSegmenter()

    gui = SE_GUI(segmenter.run)


# '''Code Based Approach, allows for inputs to be manually input into an object'''
# if __name__ == "__main__":

#     segmenter = SceneSegmenter()

#     segmenter.run(filename="./Falling.csv", 
#                   split_method="sentences",
#                   smooth="gaussian1d",
#                   plot=True)
    
