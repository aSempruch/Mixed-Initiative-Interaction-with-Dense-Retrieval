import pickle as pkl
import pandas as pd
import argparse
import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np

from util import get_arg
from util.wikitext_proc import process_line
# %%

data_path = get_arg('data_path')
experiment = get_arg('experiment')
sampling = get_arg('sampling')

# %%

with open(f'{data_path}/corpus.pkl', mode='rb') as f:
    corpus = pkl.load(f)

with open(f'{data_path}/doc_to_id_map.pkl', mode='rb') as f:
    doc_to_id_map = pkl.load(f)

train_queries = pd.read_csv('ClariQ-master/parsed/train-queries.tsv', sep='\t', header=None, index_col=0)
question_bank = pd.read_csv('ClariQ-master/parsed/question_bank.tsv', sep='\t', header=None, index_col=0)

first_path = f'{data_path}/1'
second_path = f'{data_path}/2'

for path in [first_path, second_path]:
    if not os.path.exists(path):
        os.makedirs(path)


bm25 = BM25Okapi(corpus)
corpus_len = len(corpus)

# %% First phase (request, document) triples
print('Phase 1')
with open(f'{first_path}/triples_{experiment}.jsonl', mode='w') as f:

    for question_id, question in tqdm(train_queries.itertuples(), total=train_queries.shape[0], desc=f'Constructing {sampling} triples'):
        proc_query = process_line(question)

        if sampling == 'random_negative':
            passage_tokenized = bm25.get_top_n(proc_query, corpus, n=1)[0]

            passage_id = doc_to_id_map[" ".join(passage_tokenized)]
            random_negative_passage_id = np.random.randint(0, corpus_len)
            negative_passage_id = random_negative_passage_id

        elif sampling == 'low_ranked':
            bm25_results = bm25.get_top_n(proc_query, corpus, n=100)

            passage_tokenized = bm25_results[0]
            passage_id = doc_to_id_map[" ".join(passage_tokenized)]

            negative_passage_tokenized = bm25_results[-1]
            negative_passage_id = doc_to_id_map[" ".join(negative_passage_tokenized)]

        else:
            raise ValueError(f'Invalid sampling strategy {sampling}')

        f.write(f'[{question_id}, {passage_id}, {negative_passage_id}]\n')

# %% Second phase (document, question) triples
print('Phase 2')
with open(f'{second_path}/triples_{experiment}.jsonl', mode='w') as f:

    for question_id, question in tqdm(question_bank.itertuples(), total=question_bank.shape[0], desc='Constructing random negative triples'):
        proc_query = process_line(question)
        passage_tokenized = bm25.get_top_n(proc_query, corpus, n=1)[0]

        passage_id = doc_to_id_map[" ".join(passage_tokenized)]
        random_negative_question_id = question_bank.sample(n=1).index[0]

        f.write(f'[{passage_id}, {question_id}, {random_negative_question_id}]\n')

