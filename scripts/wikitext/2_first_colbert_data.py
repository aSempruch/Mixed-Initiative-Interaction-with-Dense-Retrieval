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
question_bank = pd.read_csv('ClariQ-master/parsed/question_bank.tsv', sep='\t', header=None, index_col=0)

first_path = f'{data_path}/1'
second_path = f'{data_path}/2'

for path in [first_path, second_path]:
    if not os.path.exists(path):
        os.makedirs(path)


bm25 = BM25Okapi(corpus)
corpus_len = len(corpus)

# First phase (request, document) triples
for idx, (path, queries) in enumerate([(first_path, train_queries), (second_path, question_bank)]):
    print(f'Phase {idx+1}')
    with open(f'{path}/triples_random_negatives.jsonl', mode='w') as f:

        for query_id, query in tqdm(queries.itertuples(), total=queries.shape[0], desc='Constructing random negative triples'):
            proc_query = process_line(query)
            positive_tokenized = bm25.get_top_n(proc_query, corpus, n=1)[0]

            # TODO: this should be sampling from corpus for first phase and question_bank for second phase
            positive_id = doc_to_id_map[" ".join(positive_tokenized)]
            random_negative_id = np.random.randint(0, corpus_len)

            f.write(f'[{query_id}, {positive_id}, {random_negative_id}]\n')
