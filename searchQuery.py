import pandas as pd
from nlppreprocess import NLP
import nltk
#nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet

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

if __name__ == "__main__":
    # Query="Unexpected Heartbreak,<p>"
    Query=input("Enter your query:")
    df1 = pd.DataFrame({"query":[Query]})
    originalQueryList = preprocessQuery(df1)

    Query = Query.split(" ")
    syns = []
    for word in Query:
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                syns.append(l.name())

    synQuery = Query + syns

    df = pd.DataFrame({"query": [synQuery]})
    queryList = preprocessQuery(df)

    print("Query List: " + str(queryList))
    inverted_index = inv.load_inverted_index()
    track_list = inv.load_tracks()
    stats = {
        'inverted_index': inverted_index,
        'track_list': track_list
    }
    result=[]
    for word in queryList:
        wordDocs=[]
        if word in inverted_index.keys():
            wordDocs=inverted_index.get(word)
            result.append(wordDocs)

    x=set.intersection(*[set(x) for x in result])
    d = result[0].keys()
    ordered_results = r.rank_results(stats, queryList, originalQueryList, list(d), ranker="pln")

    print("-------------------------------")
    for idx, val in enumerate(ordered_results):
        track, artist = inv.get_track_det(val[1], track_list)
        print(str(idx+1) + ': ' + 'Score: ' + str(val[0]))
        print(track + " by " + artist)
        print("")
        if idx == 9:
            break

    # recommended_tracks = list()
    # for doc in list(x):
    #     track, artist = inv.get_track_det(doc, track_list)
    #     recommended_tracks.append(track)
    # print(recommended_tracks)




    # for i in range(2):
    #     print(i)
    #     l1,l2=result[:i+2]

    #     print(l1)
    #     print(l2)
    #     res=[item for item in l1 if item in l2]
    #     print("Intersection",res)
    #     # think if one word is not present
