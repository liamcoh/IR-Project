# IR-Project

This Repository is part of the course Information Retrieval BGU 2022.
Here you can find our solution to the Final Project, Wikipedia Search Engine

# Structure of Repository:

## Algorithm
search_frontend.py

search_backend.py

Both include out optimized solution build with API run on GCP.

## Evaluation
Eval code used to assess our Retrieval Methods

## Retrieval Methods
Experiments we tried during out testing, TFIDF Cosine Similarity and BM25, more of which in our report

sub directories of different experiments include source code and evaluation code with outputs of different messures

## GCP
Folder includes all calculations we run on GCP and saved to our Bucket.

Sizes of Bucket File: code and text form of requested human readable sizes for each file in our bucket

Document Length and Title: two dicts of doc_id keys, and doc Length/Title

Pageranks: dict of page rank for each doc

Pageviews: dict of page views for each doc

TFIDF Vector lengths: normalized and unormalized dict for each doc of cosine similiarity denominator doc part, 
called vector length as seen here, https://en.wikipedia.org/wiki/Cosine_similarity

Indexes: various Inverted Indexes creations
- Body_Index: on body text, with tf score for each doc
- Body_BM25_stem: on body text, dict consist of stemmed tokens, with bm25 score for each doc
- Body_2gram_stem: on body text, dict consist of 2 sized, stemmed, shingles, with bm25 score for each doc
- Title_Index: on title text, with tf score for each doc
- Title_stem: on title text, dict consist of stemmed tokens, with tf score for each doc
- Anchor_Index_Links: on anchor text, dict consist of tokens, posting of links, for each doc multiple links were saved by the form of (original doc id, linked doc id)
