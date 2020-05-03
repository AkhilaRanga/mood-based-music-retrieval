import pandas as pd
from nlppreprocess import NLP
import nltk  
nltk.download('wordnet')
import process_Data as p
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
    # Query="Happy and Songs,<p>"

    d = [[1,2,3,4], [2,3,4], [3,4,5,6,7]]  
    x=set.intersection(*[set(x) for x in d]) 

    
    print(list(x))

    Query="Unexpected Heartbreak,<p>"
    df = pd.DataFrame({"query":[Query]})
    queryList = preprocessQuery(df) 
    data_df = p.load_json("data/album_reviews2.json")
    data_df = p.preprocess(data_df)

    inverted_index = p.create_inverted_idx(data_df)
    result=[]
    for word in queryList:
        wordDocs=[]
        if word in inverted_index:
            print(word)
            wordDocs=inverted_index.get(word)
            
            result.append(wordDocs)

    #need some kind of hardcoding
    print(result)
    x=set.intersection(*[set(x) for x in result]) 

    
    print("Intersection :",list(x))

     


    # for i in range(2):
    #     print(i)
    #     l1,l2=result[:i+2]
        
    #     print(l1)
    #     print(l2)
    #     res=[item for item in l1 if item in l2] 
    #     print("Intersection",res)
    #     # think if one word is not present 
    