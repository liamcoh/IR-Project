from flask import Flask, request, jsonify
import re
import math
import numpy as np
import pickle
import pandas as pd
import gzip
import nltk
import requests
import operator
from search_backend import *
from time import time
from collections import Counter, defaultdict
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import ngrams

nltk.download('stopwords')

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
stemmer = PorterStemmer()

# DT holds all the Titles for doc_ids
with open("dt/dt.pkl", 'rb') as f:
    dt = pickle.load(f)

with gzip.open('pr/part-00000-ec0fe8a8-087f-4281-9ee1-c57beba2e1eb-c000.csv.gz') as f:
    pr = pd.read_csv(f, header=None)

pr_dict = pr.set_index(0).to_dict()[1]
pr_highest = pr[1].max()
pr_dict = {k: v/pr_highest for k, v in pr_dict.items()}

# with open("pv/pageviews-202108-user.pkl", 'rb') as f:
#      pv = pickle.load(f)
#
with open("postings_gcp_title_stem/index.pkl", 'rb') as f:
    index_title_stem = pickle.load(f)

with open("postings_gcp_title/index.pkl", 'rb') as f:
    index_title = pickle.load(f)

with open("postings_gcp_body_stem/index.pkl", 'rb') as f:
    index_body_stem = pickle.load(f)

with open("postings_gcp_body_bm25_stem/index.pkl", 'rb') as f:
    index_body_bm25_stem = pickle.load(f)

with open("postings_gcp_2gram_body/index.pkl", 'rb') as f:
    index_2gram_body = pickle.load(f)

with open("postings_gcp_anchor_links/index.pkl", 'rb') as f:
    index_anchor_links = pickle.load(f)


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # Tokenize the query
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens = [stemmer.stem(t) for t in tokens if t not in all_stopwords]
    ngrams_tokens = []
    try:
        for ngram in list(ngrams(tokens, 2)):
            ngrams_tokens.append(ngram[0] + " " + ngram[1])
    except:
        pass

    pls = {}
    plsbody = {}
    plstitle = {}

    # Reading all posting lists we are going to use
    for term in np.unique(tokens):
        if term in index_title_stem.df.keys():
            tmp = read(index_body_bm25_stem, term, "postings_gcp_body_bm25_stem/")
            prec = np.percentile([bm for doc_id, bm in tmp], 50)
            tmp = {doc_id: bm for doc_id, bm in tmp if bm > prec}
            plsbody[term] = tmp
            plstitle[term] = read(index_title_stem, term, "postings_gcp_title_stem/")

    for term in np.unique(ngrams_tokens):
        if term in index_2gram_body.df.keys():
            pls[term] = dict(read(index_2gram_body, term, "postings_gcp_2gram_body/"))

    pls_size = 0
    max_body = 1
    for i in pls:
        pls_size = pls_size + len(i)

    # Filter by getting ngrams first
    if len(ngrams_tokens) > 0 and pls_size > 0:
        candidates = get_candidate_documents(ngrams_tokens, index_2gram_body, pls)
        res_score1 = sorted([(doc_id, bm25_score_calc(ngrams_tokens, doc_id, index_2gram_body, pls, 2.1, 0.1)/1000)
                            for doc_id in candidates], key=lambda x: x[1], reverse=True)

        # Get remaining by intersection
        if len(res_score1) < 100:
            candidates = get_candidate_documents_intersection(tokens, plsbody)
            res_score1 += sorted([(doc_id, bm25_score(tokens, doc_id, index_body_bm25_stem, plsbody) / 1000)
                                 for doc_id in candidates], key=lambda x: x[1], reverse=True)[:100-len(res_score1)]

    else:
        # Queries with ngram with no posting lists
        candidates = get_candidate_documents_intersection(tokens, plsbody)
        res_score1 = sorted([(doc_id, bm25_score(tokens, doc_id, index_body_bm25_stem, plsbody)/1000)
                            for doc_id in candidates], key=lambda x: x[1], reverse=True)

    if len(res_score1) < 100:
        # Single token queries
        candidates = get_candidate_documents(tokens, index_body_bm25_stem, plsbody)
        res_score1 += sorted([(doc_id, bm25_score(tokens, doc_id, index_body_bm25_stem, plsbody) / 1000)
                              for doc_id in candidates], key=lambda x: x[1], reverse=True)[
                      :100 - len(res_score1)]

    # Normalize bm25 score for better manipulation in merging of results
    if len(res_score1) != 0:
        max_body = max(res_score1, key=lambda x: x[1])[1]
    res_score1 = {doc_id: bm25 / max_body for doc_id, bm25 in res_score1}

    # Get Title score by unique tokens in title
    dict_results = {}
    for token in np.unique(tokens):
        if token not in index_title_stem.df.keys():
            continue
        for doc_id, tf in plstitle[token]:
            if doc_id in dict_results:
                dict_results[doc_id] = dict_results[doc_id] + 1
            else:
                dict_results[doc_id] = 1

    # Normalize Title score 0 to 1
    max_title = max(dict_results.items(), key=operator.itemgetter(1))[1]
    res_score2 = {doc_id: score / max_title for doc_id, score in dict_results.items()}

    res_score = merge_results_pr(res_score1, res_score2, pr_dict, w1=0.5, w2=0.3, w3=0.4, N=100)

    # Assign Document Title to corresponding ID
    for doc_id, score in res_score:
        if doc_id not in dt.keys():
            res.append((doc_id, 'None'))
        else:
            res.append((doc_id, dt[doc_id]))

    return jsonify(res)


@app.route("/search_body")
def search_body():

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # Tokenize the query
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens = [stemmer.stem(t) for t in tokens if t not in all_stopwords]
    # TFIDF Cos sim calc
    q_vec = generate_query_tfidf_vector(tokens, index_body_stem)
    tfidf = generate_document_tfidf_matrix(tokens, index_body_stem)
    cos_sim = cosine_similarity(tfidf, q_vec)

    # Sort Documents by Cos sim score and retrieve top 100 only
    res_score = sorted([(doc_id, score) for doc_id, score in cos_sim.items()], key=lambda x: x[1],reverse=True)[:100]
    if len(res_score) < 100:
        tfidf = generate_document_tfidf_matrix_top100(tokens, index_body_stem, res_score)
        cos_sim = cosine_similarity(tfidf, q_vec)
        remaining_res = sorted([(doc_id, score) for doc_id, score in cos_sim.items()], key=lambda x: x[1], reverse=True)
        res_score = res_score + remaining_res[:100-len(res_score)]
    # Assign Document Title to corresponding ID
    for doc_id, score in res_score:
        if doc_id not in dt.keys():
            res.append((doc_id, 'None'))
        else:
            res.append((doc_id, dt[doc_id]))

    return jsonify(res)


@app.route("/search_title")
def search_title():

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # Tokenize the query
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens = [t for t in tokens if t not in all_stopwords]

    dict_results = {}
    for token in np.unique(tokens):
        if token not in index_title_stem.df.keys():
            continue
        # Get posting list for token
        posting = read(index_title, token, "postings_gcp_title/")
        for doc_id, tf in posting:
            if doc_id in dict_results:
                dict_results[doc_id] = dict_results[doc_id] + 1
            else:
                dict_results[doc_id] = 1

    # Sort Documents by number unique of tokens in doc
    list_of_dict = sorted([(doc_id, score) for doc_id, score in dict_results.items()], key=lambda x: x[1], reverse=True)

    # Assign Document Title to corresponding ID
    for doc_id, score in list_of_dict:
        if doc_id not in dt.keys():
            res.append((doc_id, 'None'))
        else:
            res.append((doc_id, dt[doc_id]))

    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # Tokenize the query
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens = [t for t in tokens if t not in all_stopwords]

    dict_results = {}
    for token in np.unique(tokens):
        added = []
        if token not in index_anchor_links.df.keys():
            continue
        # Get posting list for token. postings in the form of list of tuples, each tuple (doc_id, link_doc_id)
        # each doc_id can appear multiple times, depending on the number of links in it.
        posting = read(index_anchor_links, token, "postings_gcp_anchor_links/")
        for doc_id, link_doc_id in posting:
            if link_doc_id in dict_results:
                if link_doc_id not in added:
                    dict_results[link_doc_id] = dict_results[link_doc_id] + 1
            else:
                added.append(link_doc_id)
                dict_results[link_doc_id] = 1

    # Sort Documents by number of unique tokens in doc
    list_of_dict = sorted([(link_doc_id, score) for link_doc_id, score in dict_results.items()], key=lambda x: x[1], reverse=True)

    # Assign Document Title to corresponding ID
    for link_doc_id, score in list_of_dict:
        if link_doc_id not in dt.keys():
            res.append((link_doc_id, 'None'))
        else:
            res.append((link_doc_id, dt[link_doc_id]))

    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    # Retrieve page rank from pr data structure, unsorted
    res = [pr.loc[pr[0] == doc_id, 1].iloc[0] for doc_id in wiki_ids if doc_id in pr[0].values]

    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    # Retrieve page views from pv data structure, unsorted
    res = [pv[doc_id] for doc_id in wiki_ids if doc_id in pv.keys()]

    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
