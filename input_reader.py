import nltk
import csv

#TODO - implement "tokens_min" and "tokens_exact". Implement conversion method to allow them to work with csv files

class InputReader():
    def __init__(self):
        nltk.download('punkt')

    def read(self, filename, split_method="sentences", split_len=50):
        if filename.endswith('.txt'):
            if split_method == "sentences":
                text = open(filename, "r", encoding="utf8").read()
                sentences = nltk.sent_tokenize(text)
                ground_truth = None

            elif split_method == "tokens_min" or "token" in split_method:
                text = open(filename, "r", encoding="utf8").read()
                sentences_raw = nltk.sent_tokenize(text)
                ground_truth = None

                sentences = []
                current_group = []
                current_len = 0

                for sent in sentences_raw:
                    current_len += len(nltk.word_tokenize(sent))
                    current_group.append(sent)
                    if current_len > split_len:
                        sentences.append(' '.join(current_group))
                        current_group = []
                        current_len = 0
                
                if len(current_group) > 0: 
                    sentences.append(' '.join(current_group))

        elif filename.endswith('.csv'):
            # Lists to store results
            ground_truth = []
            sentences = []

            # Read the CSV file
            with open(filename, newline='', encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                if split_method == "sentences":
                    for index, row in enumerate(reader):
                        # Check the first column for '1'
                        if row[0] == '1':
                            ground_truth.append(index)
                        # Append the sentence from the second column
                        sentences.append(row[1])

                if split_method == "tokens_exact":
                    print("tokens_exact not yet implemented. Defaulting to tokens_min instead")
                    split_method = "tokens_min"

                if split_method == "tokens_min" or "token" in split_method:
                    current_group = []
                    current_len = 0
                    current_scene_val = 0
                    for index, row in enumerate(reader):
                        # Check the first column for '1'
                        if row[0] == '1':
                            current_scene_val = 1
                        # Append the sentence from the second column
                        current_group.append(row[1])
                        current_len += len(nltk.word_tokenize(row[1]))

                        if current_len > split_len:
                            if current_scene_val == 1:
                                ground_truth.append(len(sentences))
                                current_scene_val = 0
                            current_len = 0
                            sentences.append(' '.join(current_group))
                            current_group = []

                    if len(current_group) > 0:
                        if current_scene_val == 1:
                                ground_truth.append(len(sentences))
                                current_scene_val = 0
                        sentences.append(' '.join(current_group))
                            
                            

        num_tokens = self.count_tokens(sentences)

        return sentences, ground_truth, num_tokens


    def count_tokens(self, sentences):
        total_tokens = 0
        for sentence in sentences:
            # Tokenize the sentence into words
            tokens = nltk.word_tokenize(sentence)
            # Count the tokens in this sentence
            total_tokens += len(tokens)
        return total_tokens
