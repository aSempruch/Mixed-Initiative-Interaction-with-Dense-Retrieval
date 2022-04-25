import pickle as pkl
import pandas as pd
import argparse
import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np

from util.wikitext_proc import process_line
# %%

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_path', required=True)

args = vars(arg_parser.parse_args())

data_path = args['data_path']

# %%

with open(f'{data_path}/corpus.pkl', mode='rb') as f:
    corpus = pkl.load(f)

with open(f'{data_path}/doc_to_id_map.pkl', mode='rb') as f:
    doc_to_id_map = pkl.load(f)

train_queries = pd.read_csv('ClariQ-master/parsed/train-queries.tsv', sep='\t', header=None, index_col=0)

first_path = f'{data_path}/1'
if not os.path.exists(first_path):
    os.makedirs(first_path)


with open(f'{first_path}/triples_random_negatives.jsonl', mode='w') as f:
    bm25 = BM25Okapi(corpus)
    corpus_len = len(corpus)

    for query_id, query in tqdm(train_queries.itertuples(), total=train_queries.shape[0], desc='Constructing random negative triples'):
        proc_query = process_line(query)
        positive_tokenized = bm25.get_top_n(proc_query, corpus, n=1)[0]

        positive_id = doc_to_id_map[" ".join(positive_tokenized)]
        random_negative_id = np.random.randint(0, corpus_len)

        f.write(f'[{query_id}, {positive_id}, {random_negative_id}]\n')
