import pandas as pd
from nlppreprocess import NLP
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('popular', quiet=True)
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
    result = []
    for group in originalQueryList:
        docs=[]
        for word in group:
            if word in inverted_index.keys():
                Docs=inverted_index.get(word)
                docs += Docs.keys()
        result.append(docs)
    return result

def queryExpansion(query_terms):
    synonyms = []

    for term in query_terms:
        term_syn = []
        if len(term) > 1 and term.isalpha():
            for syn in wordnet.synsets(term):
                for l in syn.lemmas():
                    if l.name() not in term_syn and '_' not in l.name():
                        term_syn.append(l.name())
            
        synonyms.append(term_syn)

    return synonyms

if __name__ == "__main__":
    # Query="Unexpected Heartbreak,<p>"
    # Query=input("Enter your query:")
    Query="Happy Dreamy,<p>"
    tokens = word_tokenize(Query.lower())
    print("After tokenize: " + str(tokens))

    # POS_TAG was problematic improve later
    # word_tag_pairs = nltk.pos_tag(Query)
    # print(word_tag_pairs)
    # query_terms = [a for (a, b) in word_tag_pairs if b == 'JJ']


    expanded = queryExpansion(tokens)

    df = pd.DataFrame({"query":[Query]})
    tokens = preprocessQuery(df)
    originalQueryList = tokens

    full_query = []

    for words in expanded:
        if len(words) == 0:
            continue
        Query = ''
        for word in words:
            Query = Query + ' ' + word
        df = pd.DataFrame({"query":[Query]})
        full_terms = preprocessQuery(df)
        full_query.append(full_terms)

    print(Query)

    print("Processed: " + str(full_query))
    
    inverted_index = inv.load_inverted_index('inverted_index.pkl')
    inverted_index_l = inv.load_inverted_index('inverted_index_lyric.pkl')
    track_list = inv.load_tracks('track_dict.pkl')
    track_list_l = inv.load_tracks('lyric_dict.pkl')

    critic_docs = retrievDocs(full_query, inverted_index)
    lyric_docs = retrievDocs(full_query, inverted_index_l)

    # print(critic_docs)
    # print(lyric_docs)

    queryList = full_query

    critic_docs_x = set.intersection(*[set(x) for x in critic_docs])
    lyric_docs_x = set.intersection(*[set(x) for x in lyric_docs])

    global_docs = critic_docs_x.union(lyric_docs_x)

    print(global_docs)

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

    # ordered by track_id
    ordered_results_review = r.rank_results(stats, queryList, originalQueryList, list(global_docs), ranker="pln", discount_factor=0.75)
    ordered_results_lyrics = r.rank_results(stats_lyrics, queryList, originalQueryList, list(global_docs), ranker="pln", discount_factor=0.75)

    k = 0.25 # lyrics will have 1/4 influence on score

    # Combine results using interpolation
    combined_results = []
    for idx, pair in enumerate(ordered_results_review):
        score = pair[0] * (1-k) + ordered_results_lyrics[idx][0] * k
        combined_results.append((score, pair[1]))

    # order results by score
    ordered_results = sorted(combined_results, reverse=True, key=lambda x: x[0])

    print("-------------------------------")
    for idx, val in enumerate(ordered_results):
        track, artist = inv.get_track_det(val[1], track_list)
        print(str(idx+1) + ': ' + 'Score: ' + str(val[0]))
        print(track + " by " + artist)
        print("")
        if idx == 9:
            break

    # # recommended_tracks = list()
    # # for doc in list(x):
    # #     track, artist = inv.get_track_det(doc, track_list)
    # #     recommended_tracks.append(track)
    # # print(recommended_tracks)

