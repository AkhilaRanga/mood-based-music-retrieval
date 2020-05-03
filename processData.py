# This file pre-processes the json file and creates an inverted index
# imports
import pandas as pd
from nlppreprocess import NLP
import nltk
nltk.download('wordnet')
from pickle import dump, load

# load the crawled json file
def load_json(filename):
    return pd.read_json(filename)

# apply stemming
def stemming(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(word) for word in w_tokenizer.tokenize(text)]

# preprocess data
def preprocess(df, rev_type):
    nlp = NLP()
    if(rev_type=="album"):
        df['abstract'] = df['abstract'].apply(nlp.process)
        df['abstract'] = df['abstract'].str.lower()
        df['abstract'] = df.abstract.apply(stemming)
    df['body'] = df['body'].apply(nlp.process)
    df['body'] = df['body'].str.lower()
    df['body'] = df.body.apply(stemming)
    return df

# creating a dictionary of tracks 
# Dictionary maps index to the track/album and contains length of track/album review
# Dictionary format -- {track/album index : [ length of body and/or abstract, track/album name, artist name] }
def create_track_dict(df, rev_type):
    track_dict = {}
    for index, row in df.iterrows():
        track_dict[index] = list()
        if(rev_type=="track"):
            track_dict[index].append(len(row['body']))
            track_dict[index].append(row['track-name'])
        else:
            track_dict[index].append(len(row['body'])+len(row['abstract']))
            track_dict[index].append(row['album-name'])
        track_dict[index].append(row['artist-name'])
    return track_dict

# creating inverted index
# inverted index format -- { word : [ {track/album index : word_freq}, .. ] }
def create_inverted_idx(df, rev_type):
    inverted_index = {}
    # each row of dataframe is a document
    for index, row in df.iterrows():
        # only album reviews have abstract
        # abstract
        if(rev_type=="album"):
            abs_set = set(word for word in row['abstract'])
            abs_dict = {}
            for word in row['abstract']:
                if(word not in abs_dict):
                    abs_dict[word] = 1
                else:
                    abs_dict[word] += 1
            for word in abs_dict:
                if word not in inverted_index:
                    inverted_index[word] = {}
                    inverted_index[word][index] = abs_dict[word]
                else:
                    inverted_index[word][index] = abs_dict[word]
        
        # body
        body_dict = {}
        for word in row['body']:
            if(word not in body_dict):
                body_dict[word] = 1
            else:
                body_dict[word] += 1
        for word in body_dict:
            if word not in inverted_index:
                inverted_index[word] = {}
                inverted_index[word][index] = body_dict[word]
            else:
                inverted_index[word][index] = body_dict[word]
    return inverted_index

if __name__ == "__main__":
    # json file as a dataframe
    data_df = load_json("data/track_reviews.json")
    data_df = preprocess(data_df, "track") # "album" for album_reviews
    track_dict = create_track_dict(data_df, "track")
    dump(track_dict, open('data/track_dict.pkl', 'wb'))
    print("----done----")
    inverted_index = create_inverted_idx(data_df, "track") # "album" for album_reviews
    dump(inverted_index, open('data/inverted_index.pkl', 'wb'))
    print("----done----")
    # inv = load( open('data/inverted_index.pkl', 'rb') )