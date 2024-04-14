import nltk
import csv

class InputReader():
    def __init__(self, split_method="sentences", split_len=50):
        nltk.download('punkt')
        self.split_method = split_method
        if self.split_method not in ["sentences", "tokens"]:
            raise ValueError(f"{self.split_method} is an invalid type for split_method")
        self.split_len = split_len


    def read(self, filename):

        if filename.endswith('.txt'):
            text = open(filename, "r", encoding="utf8").read()
            sentences = nltk.sent_tokenize(text)
            ground_truth = None

        elif filename.endswith('.csv'):
            # Lists to store results
            ground_truth = []
            sentences = []

            # Read the CSV file
            with open(filename, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for index, row in enumerate(reader):
                    # Check the first column for '1'
                    if row[0] == '1':
                        ground_truth.append(index)
                    # Append the sentence from the second column
                    sentences.append(row[1])

        num_tokens = self.count_tokens(sentences)

        splt_text = self.splitter(sentences)

        return splt_text, ground_truth, num_tokens
    

    def splitter(self, sentences): 
        if self.split_method == "sentences":
            return sentences
        if self.split_method == "tokens":
            #TODO - implement a method using self.split_len
            print("Token Split not implemented")
            return sentences


    def count_tokens(self, sentences):
        total_tokens = 0
        for sentence in sentences:
            # Tokenize the sentence into words
            tokens = nltk.word_tokenize(sentence)
            # Count the tokens in this sentence
            total_tokens += len(tokens)
        return total_tokens
