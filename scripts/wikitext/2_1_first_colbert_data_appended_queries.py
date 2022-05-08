import pickle as pkl
import pandas as pd
import argparse
import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

from util import get_arg, run_config_from_args
from util.wikitext_proc import process_line
# %%

data_path = get_arg('data_path')
experiment = get_arg('experiment')
index_name = get_arg('colbert_index_name')

# %%


with open(f'{data_path}/corpus.pkl', mode='rb') as f:
    corpus = pkl.load(f)

with open(f'{data_path}/doc_to_id_map.pkl', mode='rb') as f:
    doc_to_id_map = pkl.load(f)

train_data = pd.read_csv(f'ClariQ-master/data/train.tsv', sep="\t")
train_queries = pd.read_csv('ClariQ-master/parsed/train-queries.tsv', sep='\t', header=None, index_col=0)
question_bank = pd.read_csv('ClariQ-master/data/question_bank.tsv', sep="\t", index_col="question_id")
question_bank.dropna(inplace=True)
question_id_table = {q_idx: idx for idx, (q_idx, _) in enumerate(question_bank.iterrows())}

first_path = f'{data_path}/1'
second_path = f'{data_path}/2'

for path in [second_path]:
    if not os.path.exists(path):
        os.makedirs(path)


# bm25 = BM25Okapi(corpus)
corpus_len = len(corpus)

run_config = run_config_from_args()

# TODO: run config args
with Run().context(run_config):
    searcher = Searcher(index=index_name)

with open(f'{second_path}/triples_{experiment}.jsonl', mode='w') as f:
    append_queries_dict = dict()
    for idx, (request_id, request) in enumerate(tqdm(train_queries.itertuples(), total=train_queries.shape[0], desc='Constructing random negative triples')):
        result = searcher.search(request, k=1)  # TODO: parametrize k
        # TODO: create new queries file that has concatenated queries to passages
        top_document_id = result[0][0]

        positives = train_data[
            train_data.initial_request == request
            ]['question_id'].map(question_id_table).dropna().astype(int)
        negatives = train_data[
            (train_data['initial_request'] != request)
            & (~train_data['question_id'].isin(positives))
            ]['question_id'].map(question_id_table).dropna().astype(int)

        # passage_id = doc_to_id_map[" ".join(passage_tokenized)]
        # random_negative_question_id = question_bank.sample(n=1).index[0]

        for positive in positives:
            random_negative = negatives.sample(1).values[0]
            f.write(f'[{top_document_id}, {positive}, {random_negative}]\n')

        append_queries_dict[top_document_id] = " ".join([request, " ".join(corpus[top_document_id])])

queries_appended_docs_df = pd.DataFrame.from_dict(append_queries_dict, orient='index')
queries_appended_docs_df.to_csv(
    f'{data_path}/collection_appended.tsv',
    sep='\t',
    header=None,
    doublequote=False
)

# %% Second phase (document, question) triples
# print('Phase 2')
# with open(f'{second_path}/triples_{experiment}.jsonl', mode='w') as f:
#
#     for question_id, question in tqdm(question_bank.itertuples(), total=question_bank.shape[0], desc='Constructing random negative triples'):
#         proc_query = process_line(question)
#         passage_tokenized = bm25.get_top_n(proc_query, corpus, n=1)[0]
#
#         passage_id = doc_to_id_map[" ".join(passage_tokenized)]
#         random_negative_question_id = question_bank.sample(n=1).index[0]
#
#         f.write(f'[{passage_id}, {question_id}, {random_negative_question_id}]\n')

    # TODO: make a separate script where you query first colbert model with initial request, then concat the query with the returned doc and add it to queries file
    # train the model on these triples then
