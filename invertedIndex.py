# load tracks and inverted index and use the functions here
import processData as p
from pickle import load

# load tracks dictionary
def load_tracks():
    return load( open('data/track_dict.pkl', 'rb') )

# load inverted index
def load_inverted_index():
    return load( open('data/inverted_index.pkl', 'rb') )

# get length of a document(track/album review) given document index
def doc_length(doc_index, load_tracks):
    return load_tracks[doc_index][0]

# get document frequency (number of track/album reviews in which the term appears)
def doc_freq(word, inv_idx):
    return len(inv_idx[word])

# get track/album name and artist name given the document index
def get_track_det(doc_index, load_tracks):
    return load_tracks[doc_index][1], load_tracks[doc_index][2]

# get term frequency of a word in a document(track/album review)
def term_freq(word, doc_index, inv_idx):
    return inv_idx[word][doc_index]

# get number of documents(tracks/albums)
def num_doc(load_tracks):
    return len(load_tracks)

# get the list of documents(tracks/albums)
def get_track_list(load_tracks):
    return load_tracks.keys()