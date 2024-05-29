from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from Embedder import *

'''This class reads in the data from a file, then creates train/test/validation splits accordingly '''
class DataReader():
    def __init__(self, device=None):
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def prepare_data_loader(self, df, embeddings, batch_size=16):
        texts = df['text'].tolist()
        labels = torch.tensor(df['isTransition'].values, dtype=torch.float)
        return DataLoader(TextDataset(texts, labels, embeddings), batch_size=batch_size, shuffle=True)

    def generate_loaders(self, embedder, filename='example_dataset_with_controls.csv', 
                         train_split=0.6, test_split=0.2, val_split=0.2):
        
        if train_split + test_split + val_split != 1:
            raise ValueError(f"Train/Test/Validation values must add up to one. {train_split + test_split + val_split} is not valid.")
        
        df = pd.read_csv(filename)
        emb_filename = f"./Data/emb_{os.path.basename(filename)}"

        if os.path.exists(emb_filename):
            print(f"Loading embeddings from {emb_filename}")
            embeddings_df = pd.read_csv(emb_filename)
            embeddings = embeddings_df.iloc[:, :].values
        else:
            print(f"Creating embeddings file at {emb_filename}")
            embeddings = embedder.generate_embeddings(df['text'].tolist())
            embeddings_df = pd.DataFrame(embeddings)
            embeddings_df.to_csv(emb_filename, index=False)

        df['embeddings'] = embeddings.tolist()

        train_data, temp_data = train_test_split(df, test_size=(1-train_split), random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=(test_split / (val_split + test_split)), random_state=42)

        train_loader = self.prepare_data_loader(train_data, train_data['embeddings'].tolist())
        test_loader = self.prepare_data_loader(test_data, test_data['embeddings'].tolist())
        val_loader = self.prepare_data_loader(val_data, val_data['embeddings'].tolist())

        return train_loader, test_loader, val_loader


class TextDataset(Dataset):
    def __init__(self, texts, labels, embeddings):
        self.texts = texts
        self.labels = labels
        self.embeddings = torch.tensor(embeddings, dtype=torch.float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.embeddings[idx], self.labels[idx]
