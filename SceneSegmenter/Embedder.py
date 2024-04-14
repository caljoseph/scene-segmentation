from sentence_transformers import SentenceTransformer

#TODO - add in more options 
class Embedder():
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    #generates embeddings from a txt file
    def generate_embeddings(self, split_text): #split_text is a list of strings
        embeddings = self.model.encode(split_text)
        return embeddings