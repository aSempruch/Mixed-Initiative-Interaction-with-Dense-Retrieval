import faiss
import os
from util import get_arg, colbert_config_from_args, run_config_from_args
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher, Trainer
from colbert.infra import Run

triples = get_arg('colbert_triples')
queries = get_arg('colbert_queries')
collection = get_arg('colbert_collection')

if __name__ == '__main__':
    with Run().context(run_config_from_args(['colbert_name'])):
        trainer = Trainer(
            triples,
            queries,
            collection,
            config=colbert_config_from_args(['colbert_name'])
        )
        trainer.train()
