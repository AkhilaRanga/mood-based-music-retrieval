import pandas as pd
from nlppreprocess import NLP
import nltk  
nltk.download('wordnet')
import processData as p
import invertedIndex as inv
# def search(String query):
#     

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
    Query="Unexpected Heartbreak,<p>"
    # Query=input("Enter your query:")
    df = pd.DataFrame({"query":[Query]})
    queryList = preprocessQuery(df) 
    inverted_index = inv.load_inverted_index()
    track_list = inv.load_tracks()
    result=[]
    for word in queryList:
        wordDocs=[]
        if word in inverted_index.keys():
            wordDocs=inverted_index.get(word)
            result.append(wordDocs)

    x=set.intersection(*[set(x) for x in result])
    recommended_tracks = list()
    for doc in list(x):
        track, artist = inv.get_track_det(doc, track_list)
        recommended_tracks.append(track)
    print(recommended_tracks)

     


    # for i in range(2):
    #     print(i)
    #     l1,l2=result[:i+2]
        
    #     print(l1)
    #     print(l2)
    #     res=[item for item in l1 if item in l2] 
    #     print("Intersection",res)
    #     # think if one word is not present 
    