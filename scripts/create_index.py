import faiss
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from util import get_arg, run_config_from_args, colbert_config_from_args
import pandas as pd
import os

# %%

collection = get_arg('colbert_collection')
checkpoint = get_arg('colbert_checkpoint')
index_name = get_arg('colbert_index_name')

run_config = run_config_from_args()
colbert_config = colbert_config_from_args()

# %%

if __name__ == '__main__':
    with Run().context(run_config):

        indexer = Indexer(checkpoint=checkpoint, config=colbert_config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
