from flask import Flask, request, jsonify
import re
import math
import numpy as np
import pickle
import pandas as pd
import gzip
import nltk
import requests
from search_backend import *
from time import time
from collections import Counter, defaultdict
from nltk.stem.porter import *
from nltk.corpus import stopwords

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

with open("pv/pageviews-202108-user.pkl", 'rb') as f:
     pv = pickle.load(f)

with open("postings_gcp_title_sample/index.pkl", 'rb') as f:
    index_title_sample = pickle.load(f)

#with open("postings_gcp_body_sample/index.pkl", 'rb') as f:
#    index_body_tf_sample = pickle.load(f)

with open("postings_gcp_body_stem_sample/index.pkl", 'rb') as f:
    index_body_stem_sample = pickle.load(f)

with open("postings_gcp_anchor_links_sample/index.pkl", 'rb') as f:
    index_anchor_links_sample = pickle.load(f)


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
    # TFIDF Cos sim calc
    q_vec = generate_query_tfidf_vector(tokens, index_body_stem_sample)
    tfidf = generate_document_tfidf_matrix(tokens, index_body_stem_sample)
    cos_sim = cosine_similarity(tfidf, q_vec)

    # Sort Documents by Cos sim score and retrieve top 100 only
    res_score = sorted([(doc_id, score) for doc_id, score in cos_sim.items()], key=lambda x: x[1],reverse=True)[:100]
    if len(res_score) < 100:
        tfidf = generate_document_tfidf_matrix_top100(tokens, index_body_stem_sample, res_score)
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
    q_vec = generate_query_tfidf_vector(tokens, index_body_stem_sample)
    tfidf = generate_document_tfidf_matrix(tokens, index_body_stem_sample)
    cos_sim = cosine_similarity(tfidf, q_vec)

    # Sort Documents by Cos sim score and retrieve top 100 only
    res_score = sorted([(doc_id, score) for doc_id, score in cos_sim.items()], key=lambda x: x[1],reverse=True)[:100]

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
        if token not in index_title_sample.df.keys():
            continue
        # Get posting list for token
        posting = read(index_title_sample, token, "postings_gcp_title_sample/")
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
        if token not in index_anchor_links_sample.df.keys():
            continue
        # Get posting list for token. postings in the form of list of tuples, each tuple (doc_id, link_doc_id)
        # each doc_id can appear multiple times, depending on the number of links in it.
        posting = read(index_anchor_links_sample, token, "postings_gcp_anchor_links_sample/")
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
