import csv
from InputReader import InputReader
from Embedder import Embedder
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def ExtractFirstLast(filenames): #takes ground truth csv filename 
    last_sentences = []
    first_sentences = []
    full_sentences = []
    not_first = []
    last_row = None
    for filename in filenames:
        with open(filename, newline='', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for _, row in enumerate(reader):
                if row[0] == '1':
                    if last_row is not None:
                        last_sentences.append(last_row[1])
                    first_sentences.append(row[1])

                else:
                    not_first.append(row[1])

                last_row = row

                

                # Append the sentence from the second column
                full_sentences.append(row[1])

            last_sentences.append(last_row[1])

        #each key/value pair in this dictionary is a scene, with the key being the first sentence and 
        #the value being the last sentence
        paired_sentences = {first_sentences[scene] : last_sentences[scene] for scene in range(len(first_sentences))}

    return first_sentences, last_sentences, paired_sentences, full_sentences, not_first


input_reader = InputReader()
embedder = Embedder('all-MiniLM-L6-v2')

first_sentences, last_sentences, paired_sentences, full_sentences, not_first = ExtractFirstLast(["Collision_of_Worlds_5.csv", "falling.csv"])

embeddings_full = embedder.generate_embeddings(not_first)
embeddings_last = embedder.generate_embeddings(last_sentences)

# Combine the embeddings and create labels
embeddings = np.vstack((embeddings_full, embeddings_last))
labels = np.array([0] * embeddings_full.shape[0] + [1] * embeddings_last.shape[0])

# Standardize the data
scaler = StandardScaler()
embeddings_standardized = scaler.fit_transform(embeddings)

# Apply PCA
pca = PCA(n_components=2)  # Adjust the number of components as needed
principal_components = pca.fit_transform(embeddings_standardized)

# Create a DataFrame for visualization
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['Label'] = labels

# Plot the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[df_pca['Label'] == 0]['PC1'], df_pca[df_pca['Label'] == 0]['PC2'], label='Set A')
plt.scatter(df_pca[df_pca['Label'] == 1]['PC1'], df_pca[df_pca['Label'] == 1]['PC2'], label='Set B')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Sentence Embeddings')
plt.legend()
plt.show()



