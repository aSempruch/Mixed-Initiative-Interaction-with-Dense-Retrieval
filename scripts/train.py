import faiss
import os
from util import colbert_config_from_args, run_config_from_args
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher, Trainer
from colbert.infra import Run

triples = os.getenv('colbert_triples')
collection = os.getenv('colbert_collection')

if not os.getenv('colbert_name'):
    raise EnvironmentError('colbert_name not set')

if __name__ == '__main__':
    with Run().context(run_config_from_args()):
        trainer = Trainer(
            triples,
            'ClariQ-master/parsed/train-queries.tsv',
            'ClariQ-master/parsed/question_bank.tsv',
            config=colbert_config_from_args()
        )
        trainer.train()
