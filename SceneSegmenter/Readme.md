This folder contains the python implementation of the scene segmentation research. 
Note that this is also a package that can be pip installed. [Not yet implementated]
It has the following structure:

## Files
### SceneSegment.py:
The central class, draws various functionality from other files. When running the system without using the GUI, just create a single instance of it with the settings you want. 
### InputReader.py
Deals with all external files, including txt and csv inputs for both the text itself and the groundtruth. Input csv files are assumed to have a column of 0's and 1's next to a column of sentences, with 1's indicating the first sentence of a scene.
### Embedder.py
Creates embeddings from the split inputs
### SceneIdentifier.py
identifies scenes using difference measure, then applies smoothing to embeddings

## Run Instructions
- Option 1 (recommended): run the "Main.py" file, it will create a GUI for you to select options.
- Option 2: use the commented out code in "Main.py" in your own file to create a scene segmenter object.

Regardless of which method is used, a path to a valid txt or csv file must be provided. 

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