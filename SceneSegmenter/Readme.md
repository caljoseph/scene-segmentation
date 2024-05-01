This folder contains the python implementation of the scene segmentation research. It has the following structure:

## Folders
- SceneSegment.py: the central class, draws various functionality from other files. When running the system, just create a single instance of it with the settings you want. 
- InputReader.py: Deals with all external files, including txt and csv inputs for both the text itself and the groundtruth. 
- Embedder.py: Creates embeddings from the split inputs
- SceneIdentifier.py: Processes embeddings, then identifies scenes

## Options:
### Split Method: method of splitting text
 - sentences - split text on a sentence by sentence basis using NLTK
 - tokens_exact - split text into pieces of exactly k tokens
 - tokens_min - split text into sentence groups with a minimum of k tokens  

### Diff/Difference Measure: The method of determining distance between embeddings
 - pnorm - any norm, of format ("inf"/int) + "norm". e.g. 1norm, 2norm, 50norm, infnorm are all valid
 - cosine - cosine similarity

### Smooth: method of smoothing distance curve
 - Gaussian1d
 - None

### Sigma: 
 - int (between 0 and inf)