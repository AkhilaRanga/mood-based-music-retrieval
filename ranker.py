import numpy as np
import invertedIndex as inv

# Mood-based therefore assuming query is one word

# assuming stats object is a dictionary of terms with statistics regarding a term

def log_base(x, base):
    return np.log(x) / np.log(base)

def tfidf(stats):

    c_wd = stats['tf']
    N = stats['N']
    df = stats['df']

    # assuming stats object
    # need count of word W in document D c(w;d)
    # need total number of documents N
    # need document frequency df

    return (1 + log_base(c_wd, 2)) * log_base((N + 1) / df, 2)

def pln(stats):

    c_wd = stats['tf']
    n = stats['n']
    n_avg = stats['n_avg']
    s = 0.5
    N = stats['N']
    df = stats['df']
    c_wq = 1

    return (1 + np.log(1 + np.log(c_wd))) / (1 - s + s * (n/n_avg)) * c_wq * np.log((N+1) / df)

def okapi_bm25(stats):

    c_wd = stats['tf']
    N = stats['N']
    df = stats['df']
    c_wq = 1
    k_1 = 1.6 # 1.2 -> 2
    k_2 = 500 # 0 -> 1000
    b = 0.975 # 0.75 -> 1.2
    n = stats['n']
    n_avg = stats['n_avg']

    return np.log((N - df + 0.5) / (df + 0.5)) * ( ((k_1 + 1) * c_wd) / (k_1 * (1 - b + b * (n / n_avg))) ) * ( ((k_2 + 1) * (c_wq)) / (k_2 + c_wq) )


# Take input for query
# Using statistics rank the results and display in decreasing order

def rank_results(stats, query_list, original_query, results, ranker="tfidf"):
    rankers = {
        'tfidf': tfidf,
        'pln': pln,
        'bm25': okapi_bm25
    }

    ranks = [] # list of typles (score, track_id)

    for doc_id in results:

        score = 0

        for word in query_list:
            n_avg = sum([inv.doc_length(i, stats['track_list']) for i in results]) / inv.doc_freq(word, stats['inverted_index'])
            parsed_stats = {
                'tf': inv.term_freq(word, doc_id, stats['inverted_index']),
                'N': inv.num_doc(stats['track_list']),
                'df': inv.doc_freq(word, stats['inverted_index']),
                'n': inv.doc_length(doc_id, stats['track_list']),
                'n_avg': n_avg
            }

            if parsed_stats['tf'] ==0:
                score += 0
            elif word in original_query:
                score += rankers[ranker](parsed_stats)
            else:
                score += rankers[ranker](parsed_stats)*0.75

        ranks.append((score, doc_id))

    return sorted(ranks, reverse=True, key=lambda x: x[0])
