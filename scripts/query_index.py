import faiss
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from util import get_arg, run_config_from_args
import pandas as pd
import pickle as pkl
import os
import argparse
import subprocess

# %%
collection = get_arg('colbert_collection')
index_name = get_arg('colbert_index_name')

run_config = run_config_from_args()

# %%

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', choices=['clariq', 'colbert'], required=True)

    args = vars(arg_parser.parse_args())
    mode = args['mode']

    with open('ClariQ-master/parsed/question_id_map.pkl', mode='rb') as f:
        question_id_map = pkl.load(f)

    with Run().context(run_config):
        searcher = Searcher(index=index_name)

    for split in ('train', 'dev', 'test'):
        # queries = pd.read_csv('ClariQ-master/parsed/test-queries.tsv', sep='\t', index_col=0, header=None)
        queries = pd.read_csv(f'ClariQ-master/parsed/{split}-queries.tsv', sep='\t', index_col=0, header=None)
        queries_as_dict = {key:vals[0] for (key, vals) in queries.T.to_dict('list').items()}
        results = searcher.search_all(queries_as_dict, k=30)
        # results = searcher.search_all({1: 'test query'}, k=30)

        result_path = f'results/{run_config.experiment}/{index_name}'

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        with open(f'{result_path}/run_file_{split}.txt', 'w') as f:
            for topic_id, result in results.data.items():
                for passage_id, passage_rank, passage_score in result:
                    question_id = question_id_map[passage_id]
                    f.write(f'{topic_id} 0 {question_id} {passage_rank} {passage_score} {run_config.experiment}_{index_name}\n')

        eval_output = subprocess.run([
            'python',
            'ClariQ-master/src/clariq_eval_tool.py',
            '--eval_task', 'question_relevance',
            '--data_dir', '/home/asempruch/Mixed-Initiative-Interaction-with-Dense-Retrieval/ClariQ-master/data',
            '--experiment_type', split,
            '--run_file', f'{result_path}/run_file_{split}.txt',
            '--out_file', f'{result_path}/results_{split}.txt'
        ], capture_output=True)

        print(split, "\n", eval_output.stdout)
