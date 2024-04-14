This folder contains the python implementation of the scene segmentation research. It has the following structure:

- SceneSegment.py: the central class, draws various functionality from other files. When running the system, just create a single instance of it with the settings you want. 
- InputReader.py: Deals with all external files, including txt and csv inputs for both the text itself and the groundtruth. 
- InputSplitter.py: Splits the input txt into a form usable by the 
- Embedder.py: Creates embeddings from the split inputs
- SceneIdentifier.py: identifies scenes