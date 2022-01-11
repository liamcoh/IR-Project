import numpy as np
from collections import Counter, defaultdict
import math
import pickle
import pandas as pd
from itertools import combinations
from time import time

with open("dl/dl.pkl", 'rb') as f:
    DL = pickle.load(f)

# tfidf_vec_len holds all the vector length of a document pre computed
# (the lower part of Cos Sim calculation ie root sum of (tfidf Normalized) squared)
with open("tfidf_norm_vec_len/tfidf_norm_vec_len.pkl", 'rb') as f:
    tfidf_norm_vec_len = pickle.load(f)


# Read posting list for given token and index
def read(index, token, path):

    BLOCK_SIZE = 1999998
    TUPLE_SIZE = 6

    # anchor index tuple size is 8 because the tuple consist of 2 doc_ids (Integer 4 bytes each)
    if "links" in path:
        TUPLE_SIZE = 8

    # Reading bin file
    n_bytes = index.df[token] * TUPLE_SIZE
    open_files = {}
    b = []
    for f_name, offset in index.posting_locs[token]:
        if f_name not in open_files:
            open_files[f_name] = open(path + f_name, 'rb')
        f = open_files[f_name]
        f.seek(offset)
        n_read = min(n_bytes, BLOCK_SIZE - offset)
        b.append(f.read(n_read))
        n_bytes -= n_read
    b = b''.join(b)

    for f in open_files.values():
        f.close()

    # Parsing bin file
    posting_list = []
    for i in range(index.df[token]):
        doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
        tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
        posting_list.append((doc_id, tf))

    return posting_list


def generate_query_tfidf_vector(query_to_search, index):

    epsilon = .0000001
    # Create Vector for query with length equal to non zero values to avoid creating big arrays
    Q = np.zeros((1, len(np.unique(query_to_search))))
    Q = pd.DataFrame(Q)
    Q.columns = np.unique(query_to_search)
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(DL)) / (df + epsilon), 10)  # smoothing

            try:
                Q.at[0, token] = tf * idf
            except:
                pass
    return Q


def get_candidate_documents_and_scores(query_to_search, index, path):

    candidates = {}
    N = len(DL)
    postings_lists = {}

    # Reading all the query terms posting lists
    for term in np.unique(query_to_search):
        posting = read(index, term, path)
        if "title" not in path:
            posting = [(doc_id, tf) for doc_id, tf in posting if tf > 1]
        posting = {doc_id: tf for doc_id, tf in posting}
        postings_lists[term] = posting

    inter = []
    inter_num = len(np.unique(query_to_search))

    # Getting all the various intersections
    # First we find all docs that include all query tokens
    # From then we gradually decrease the number of intersecting docs required
    while len(inter) < 100:
        if inter_num < 2:
            break
        inter_ids = [doc_id for doc_id, freq_dict in inter]
        comb = combinations(np.unique(query_to_search), inter_num)
        # Comb includes all combinations in size (2 - length of query)
        for i in comb:
            comb_inter = set.intersection(*[set(postings_lists[token].keys()) for token in i])
            tmp_inter = [(doc_id, {token: postings_lists[token][doc_id] for token in i}) for doc_id in comb_inter if doc_id not in inter_ids]
            # Remove duplicates between intersections
            for doc_id, tflst in tmp_inter:
                inter.append((doc_id, tflst))

        inter_num = inter_num - 1

    # Calculate TFIDF for all Intersected docs
    for term in np.unique(query_to_search):
        if term in index.df.keys():
            tfidf = [(doc_id, (freq_dict[term] / DL[doc_id]) * math.log(N / index.df[term], 10)) for doc_id, freq_dict in
                     inter if term in freq_dict.keys()]
            for doc_id, tfidf in tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def get_candidate_documents_and_scores_top100(query_to_search, index, res, path):

    candidates = {}
    N = len(DL)
    res_ids = [doc_id for doc_id, score in res]
    for term in np.unique(query_to_search):
        if term in index.df.keys():
            # We take only top 100 docs for each term by tf value
            posting = read(index, term, path)[-100:]
            tfidf = [(doc_id, (freq / DL[doc_id]) * math.log(N / index.df[term], 10)) for doc_id, freq in
                     posting if doc_id not in res_ids]

            for doc_id, tfidf in tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix_top100(query_to_search, index, res, path):

    candidates_scores = get_candidate_documents_and_scores_top100(query_to_search, index, res, path)
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), len(np.unique(query_to_search))))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = np.unique(query_to_search)

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.at[doc_id, term] = tfidf

    return D


def generate_document_tfidf_matrix(query_to_search, index, path):

    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, path)
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), len(np.unique(query_to_search))))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = np.unique(query_to_search)

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.at[doc_id, term] = tfidf

    return D


def cosine_similarity(D, Q):

    cos_dict = {}
    for indexQ, rowQ in Q.iterrows():
        for index, row in D.iterrows():
            if index in tfidf_norm_vec_len:
                cos_sim = np.dot(row, rowQ) / (tfidf_norm_vec_len[index] * np.linalg.norm(rowQ))
                cos_dict[index] = cos_sim

    return cos_dict


def merge_results(scores1, scores2, w1=0.5, w2=0.5, N=3):

    # def merge_keys(dct):
    #     seen = set()
    #     keep = {}
    #     for k, v in dct.items():
    #         if k in seen:
    #             keep[k] = keep[k] + v
    #         else:
    #             seen.add(k)
    #             keep[k] = v
    #     return sorted([(k, v) for k, v in keep.items()], key=lambda x: x[1], reverse=True)[:N]

    scores1_w = {doc_id: score * w1 for doc_id, score in scores1}
    scores2_w = {doc_id: score * w2 for doc_id, score in scores2}

    merged = {k: scores1_w.get(k, 0) + scores2_w.get(k, 0) for k in set(scores1_w) | set(scores2_w)}
    #merged = {doc_id: score + scores2_w[doc_id] for doc_id, score in scores1_w}
    return sorted([(k, v) for k, v in merged.items()], key=lambda x: x[1], reverse=True)[:N]
    #return merge_keys(merged)