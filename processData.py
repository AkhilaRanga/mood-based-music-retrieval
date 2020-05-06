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
def preprocess(df):
    for index, row in df.iterrows():
        if(row['artist-name']!=None):
            row['body'] = row['body'].replace(row['artist-name'], '')
        row['body'] = row['body'].replace(row['track-name'], '')
    nlp = NLP()
    df['body'] = df['body'].apply(nlp.process)
    df['body'] = df['body'].str.lower()
    df['body'] = df.body.apply(stemming)
    df['lyrics'] = df['lyrics'].apply(nlp.process)
    df['lyrics'] = df['lyrics'].str.lower()
    df['lyrics'] = df.lyrics.apply(stemming)
    return df

# creating a dictionary of tracks 
# Dictionary maps index to the track and contains length of track review/lyrics
# Dictionary format -- {track index : [ length of body, track name, artist name] }
# Dictionary format -- {track index : [ length of lyrics, track name, artist name] }
def create_track_dict(df):
    track_dict = {}
    track_lyric_dict = {}
    for index, row in df.iterrows():
        # track critic reviews dict
        track_dict[index] = list()
        track_dict[index].append(len(row['body']))
        track_dict[index].append(row['track-name'])
        track_dict[index].append(row['artist-name'])
        # track lyrics dict
        track_lyric_dict[index] = list()
        track_lyric_dict[index].append(len(row['lyrics']))
        track_lyric_dict[index].append(row['track-name'])
        track_lyric_dict[index].append(row['artist-name'])
    return track_dict, track_lyric_dict

# creating inverted index
# inverted index format -- { word : [ {track index : word_freq}, .. ] }
def create_inverted_idx(df, column):
    inverted_index = {}
    # each row of dataframe is a document
    for index, row in df.iterrows():
        # body/lyrics
        text_dict = {}
        for word in row[column]:
            if(word not in text_dict):
                text_dict[word] = 1
            else:
                text_dict[word] += 1
        for word in text_dict:
            if word not in inverted_index:
                inverted_index[word] = {}
                inverted_index[word][index] = text_dict[word]
            else:
                inverted_index[word][index] = text_dict[word]
    return inverted_index

if __name__ == "__main__":
    # json file as a dataframe
    tracks_lyrics_df = load_json("data/track_review_with_lyrics.json")
    tracks_lyrics_df = preprocess(tracks_lyrics_df)
    track_dict, lyric_dict = create_track_dict(tracks_lyrics_df)
    dump(track_dict, open('data/track_dict.pkl', 'wb'))
    dump(lyric_dict, open('data/lyric_dict.pkl', 'wb'))
    print("----done----")
    inverted_index = create_inverted_idx(tracks_lyrics_df, 'body')
    inverted_index_lyric = create_inverted_idx(tracks_lyrics_df, 'lyrics')
    dump(inverted_index, open('data/inverted_index.pkl', 'wb'))
    dump(inverted_index_lyric, open('data/inverted_index_lyric.pkl', 'wb'))
    print("----done----")