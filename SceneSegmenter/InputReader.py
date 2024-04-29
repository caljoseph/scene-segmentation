import nltk
import csv

class InputReader():
    def __init__(self):
        nltk.download('punkt')

    def read(self, filename, split_method="sentences", split_len=50):
        if filename.endswith('.txt'):
            text = open(filename, "r", encoding="utf8").read()
            sentences = nltk.sent_tokenize(text)
            ground_truth = None

        elif filename.endswith('.csv'):
            # Lists to store results
            ground_truth = []
            sentences = []

            # Read the CSV file
            with open(filename, newline='', encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for index, row in enumerate(reader):
                    # Check the first column for '1'
                    if row[0] == '1':
                        ground_truth.append(index)
                    # Append the sentence from the second column
                    sentences.append(row[1])

        num_tokens = self.count_tokens(sentences)

        splt_text = self.splitter(sentences, split_method, split_len)

        return splt_text, ground_truth, num_tokens
    

    def splitter(self, sentences, split_method, split_len): 
        #Does not change how sentences are split
        if split_method == "sentences":
            return sentences
        
        #Combine sentences into blocks of min length "split_len"
        #FIXME - does not currently work with ground truths
        if split_method == "tokens": 
            combined = []  
            current_segment = [] 
            current_length = 0 

            for sentence in sentences:
                tokens = nltk.tokenize.word_tokenize(sentence)
                num_tokens = len(tokens)
                
                # If adding this sentence would make the segment at least min_length
                if current_length + num_tokens >= split_len:
                    current_segment.append(sentence)
                    combined_segment = ' '.join(current_segment)
                    combined.append(combined_segment)
                    current_segment = []  # Reset for the next segment
                    current_length = 0
                else:
                    current_segment.append(sentence)
                    current_length += num_tokens

            # Handle any remaining sentences that did not meet the min_length by themselves
            if current_segment:
                combined_segment = ' '.join(current_segment)
                combined.append(combined_segment)

            sentences = combined

            
            return sentences


    def count_tokens(self, sentences):
        total_tokens = 0
        for sentence in sentences:
            # Tokenize the sentence into words
            tokens = nltk.word_tokenize(sentence)
            # Count the tokens in this sentence
            total_tokens += len(tokens)
        return total_tokens
