from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch


class Embedder():
    def __init__(self, model_name):
        self.set_model(model_name)

    #generates embeddings from a txt file
    def generate_embeddings(self, split_text): #split_text is a list of strings
        if self.model_type == "sentence-transformer":
            embeddings = self.model.encode(split_text)
        else:
            inputs = self.tokenizer(split_text, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state

            # Extract sentence embeddings (using the [CLS] token's embeddings)
            embeddings = embeddings[:, 0, :].numpy()
        return embeddings
    
    def set_model(self, model_name):
        self.model_name = model_name
        if model_name in ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', "paraphrase-mpnet-base-v2"]:
            self.model = SentenceTransformer(model_name)
            self.model_type = "sentence-transformer"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model_type = "transformer"

        
        