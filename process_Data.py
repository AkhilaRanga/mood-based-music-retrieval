# imports
import pandas as pd
from nlppreprocess import NLP
import nltk
nltk.download('wordnet')

# load the crawled json file
def load_json(filename):
    return pd.read_json(filename)

# apply stemming
def stemming(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(word) for word in w_tokenizer.tokenize(text)]

# preprocess data
def preprocess(df):
    nlp = NLP()
    df['abstract'] = df['abstract'].apply(nlp.process)
    df['body'] = df['body'].apply(nlp.process)
    df['abstract'] = df['abstract'].str.lower()
    df['body'] = df['body'].str.lower()
    df['abstract'] = df.abstract.apply(stemming)
    df['body'] = df.body.apply(stemming)
    return df

# creating inverted index
def create_inverted_idx(df):
    inverted_index = {}
    index = 0
    for index, row in df.iterrows():
        abs_set = set(word for word in row['abstract'])
        for word in abs_set:
            if word not in inverted_index:
                inverted_index[word] = []
                inverted_index[word].append(index)
            else:
                inverted_index[word].append(index)
        body_set = set(word for word in row['body'])
        for word in body_set:
            if word not in inverted_index:
                inverted_index[word] = []
                inverted_index[word].append(index)
            else:
                inverted_index[word].append(index)
        index = index+1
    return inverted_index

if __name__ == "__main__":
    # json file as a dataframe
    data_df = load_json("data/album_reviews2.json")
    data_df = preprocess(data_df)
    inverted_index = create_inverted_idx(data_df)
    print(inverted_index)