import pandas as pd
from nlppreprocess import NLP
import nltk
#nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet
from nltk import word_tokenize 
import processData as p
import invertedIndex as inv
import ranker as r

def stemming(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(word) for word in w_tokenizer.tokenize(text)]

def preprocessQuery(df):
    nlp = NLP()
    df['query'] = df['query'].apply(nlp.process)
    df['query'] = df['query'].str.lower()
    # df['query'] = df.query.apply(stemming)
    text=df.to_string(index = False, header=False)
    tokens=[]
    tokens=stemming(text)
    return tokens

def retrievDocs(originalQueryList, inverted_index):
    synonyms = []
    result = []
    for word in originalQueryList:
        syns = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                syns.append(l.name())
        synonyms.append(syns)
        syns.append(word)
        syns = set(syns)
        wordDocs=[]
        for word in syns:
            if word in inverted_index.keys():
                Docs=inverted_index.get(word)
                wordDocs += Docs.keys()
        result.append(wordDocs)
    return result, synonyms

if __name__ == "__main__":
    # Query="Unexpected Heartbreak,<p>"
    # Query=input("Enter your query:")
    Query="Happy and Dreamy,<p>"
    Query = word_tokenize(Query.lower())
    word_tag_pairs = nltk.pos_tag(Query)
    query_terms = [a for (a, b) in word_tag_pairs if b == 'JJ']
    Query = ''
    for q in query_terms:
        Query = Query + ' ' + q
    
    df = pd.DataFrame({"query":[Query]})
    originalQueryList = preprocessQuery(df)
    print(originalQueryList)
    
    inverted_index = inv.load_inverted_index('inverted_index.pkl')
    inverted_index_l = inv.load_inverted_index('inverted_index_lyric.pkl')
    track_list = inv.load_tracks('track_dict.pkl')
    track_list_l = inv.load_tracks('lyric_dict.pkl')

    critic_docs, syns = retrievDocs(originalQueryList, inverted_index)
    lyric_docs, syns = retrievDocs(originalQueryList, inverted_index_l)

    queryList = originalQueryList + syns

    critic_docs_x = set.intersection(*[set(x) for x in critic_docs])
    lyric_docs_x = set.intersection(*[set(x) for x in lyric_docs])


    # reviews
    stats = {
        'inverted_index': inverted_index,
        'track_list': track_list
    }
    # lyrics
    stats_lyrics = {
        'inverted_index': inverted_index_l,
        'track_list': track_list_l
    }
    
    # ordered_results = r.rank_results(stats, queryList, originalQueryList, list(critic_docs_x), ranker="pln")
    # ordered_results = r.rank_results(stats, queryList, originalQueryList, list(lyric_docs_x), ranker="pln")

    # print("-------------------------------")
    # for idx, val in enumerate(ordered_results):
    #     track, artist = inv.get_track_det(val[1], track_list)
    #     print(str(idx+1) + ': ' + 'Score: ' + str(val[0]))
    #     print(track + " by " + artist)
    #     print("")
    #     if idx == 9:
    #         break

    # # recommended_tracks = list()
    # # for doc in list(x):
    # #     track, artist = inv.get_track_det(doc, track_list)
    # #     recommended_tracks.append(track)
    # # print(recommended_tracks)


