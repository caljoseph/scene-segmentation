import csv
from InputReader import InputReader
from Embedder import Embedder

def ExtractFirstLast(filename): #takes ground truth csv filename 
    last_sentences = []
    first_sentences = []
    last_row = None
    with open(filename, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for _, row in enumerate(reader):
            if row[0] == '1':
                if last_row is not None:
                    last_sentences.append(last_row[1])
                first_sentences.append(row[1])

            last_row = row

        last_sentences.append(last_row[1])

        #each key/value pair in this dictionary is a scene, with the key being the first sentence and 
        #the value being the last sentence
        paired_sentences = {first_sentences[scene] : last_sentences[scene] for scene in range(len(first_sentences))}

    return first_sentences, last_sentences, paired_sentences


input_reader = InputReader()
embedder = Embedder('all-MiniLM-L6-v2')

first_sentences, last_sentences, paired_sentences = ExtractFirstLast("Falling.csv")

sentences_full, _, _ = input_reader.read("Falling.txt")

embeddings_full = embedder.generate_embeddings(sentences_full)
embeddings_last = embedder.generate_embeddings(last_sentences)


