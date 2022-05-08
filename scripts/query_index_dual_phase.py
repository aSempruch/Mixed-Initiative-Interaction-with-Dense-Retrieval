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
from tqdm import tqdm

# %%
# collection = get_arg('colbert_collection')
first_index_name = get_arg('first_index_name')
second_index_name = get_arg('second_index_name')
data_path = get_arg('data_path')

# number of passages to concat when searching second index
concat_passages = get_arg('concat_passages', int)
appended = get_arg('appended', bool)

run_config = run_config_from_args()

# %%

with open(f'{data_path}/corpus.pkl', mode='rb') as f:
    corpus = pkl.load(f)

collection = pd.read_csv(f'{data_path}/collection.tsv', sep='\t', header=None, index_col=0)

# %%

if __name__ == '__main__':

    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('--mode', choices=['clariq', 'colbert'], required=True)
    #
    # args = vars(arg_parser.parse_args())
    # mode = args['mode']

    with open('ClariQ-master/parsed/question_id_map.pkl', mode='rb') as f:
        question_id_map = pkl.load(f)

    with Run().context(run_config):
        searchers = [Searcher(index_name) for index_name in [first_index_name, second_index_name]]

    for split in ('train', 'dev', 'test'):
        # queries = pd.read_csv('ClariQ-master/parsed/test-queries.tsv', sep='\t', index_col=0, header=None)
        queries = pd.read_csv(f'ClariQ-master/parsed/{split}-queries.tsv', sep='\t', index_col=0, header=None)
        queries_as_dict = {key: vals[0] for (key, vals) in queries.T.to_dict('list').items()}
        iter_results = searchers[0].search_all(queries_as_dict, k=concat_passages)

        results = dict()
        for topic_id, iter_result in tqdm(iter_results.data.items(), desc='Querying second index'):
            top_document_ids = list(zip(*iter_result))[0]
            # iter_queries[top_document_id] = collection.loc[top_document_id][1]
            iter_queries = collection.loc[list(top_document_ids)][1].values

            if appended:
                iter_queries[0] = " ".join([queries_as_dict[topic_id], iter_queries[0]])

            result = searchers[1].search("\n".join(iter_queries), k=30)
            results[topic_id] = list(zip(*result))
        # Query second colbert model with results of first model



        result_path = f'results/{run_config.experiment}'

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        with open(f'{result_path}/run_file_{split}.txt', 'w') as f:
            for topic_id, result in results.items():
                for passage_id, passage_rank, passage_score in result:
                    question_id = question_id_map[passage_id]
                    f.write(f'{topic_id} 0 {question_id} {passage_rank} {passage_score} {run_config.experiment}\n')

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
