This folder contains the python implementation of the scene segmentation research. It has the following structure:

- SceneSegment.py: the central class, draws various functionality from other files. When running the system, just create a single instance of it with the settings you want. 
- InputReader.py: Deals with all external files, including txt and csv inputs for both the text itself and the groundtruth. 
- Embedder.py: Creates embeddings from the split inputs
- SceneIdentifier.py: Processes embeddings, then identifies scenes


Current Issues:
Splitting according to anything other than sentences messes up with how the ground truth works, unless the ground truth is also split using the same approach. This could be rectified by using a function in the InputReader class to break it up according to tokens rather than sentences.