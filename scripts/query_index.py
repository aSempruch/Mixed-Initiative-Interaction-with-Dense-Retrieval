import faiss
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from util import get_arg, run_config_from_args
import pandas as pd
import os

# %%
collection = get_arg('colbert_collection')
index_name = get_arg('colbert_index_name')

run_config = run_config_from_args()

# %%

if __name__ == '__main__':

    with Run().context(run_config):
        searcher = Searcher(index=index_name)

    # queries = pd.read_csv('ClariQ-master/parsed/test-queries.tsv', sep='\t', index_col=0, header=None)
    queries = pd.read_csv('ClariQ-master/parsed/train-queries.tsv', sep='\t', index_col=0, header=None)
    queries_as_dict = {key:vals[0] for (key, vals) in queries.T.to_dict('list').items()}
    results = searcher.search_all(queries_as_dict, k=30)
    # results = searcher.search_all({1: 'test query'}, k=30)

    result_path = f'results/{run_config.experiment}/{index_name}'

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(f'{result_path}/run_file.txt', 'w') as f:
        for topic_id, result in results.data.items():
            for passage_id, passage_rank, passage_score in result:
                f.write(f'{topic_id} 0 {passage_id} {passage_rank} {passage_score} {run_config.experiment}_{index_name}\n')
