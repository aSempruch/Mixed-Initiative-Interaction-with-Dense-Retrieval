# mixed-initiative-interaction-dense-retrieval
Asking clarifying questions in conversational search with dense retrieval.

## Files

### `scripts/*`
This directory contains scripts for processing, training, and evaluating the indexing models.
Files under `no-corpora-baseline/` contain scripts for processing the ClariQ data necessary for training ColBERT.
Files under `wikitext/` contain scripts for processing the data with a 1 hop intermediate query comprised of WikiText
documents in mind. These files must be run first before running the files below.

* `train.py` - train ColBERT model and save checkpoint
* `create_index.py` - create ColBERT index using checkpoint
* `query_index.py` - query ColBERT index with no intermediate corpus and pass results through ClariQ eval script. 
* `query_index_dual_phase.py` - query 2 ColBERT indices with intermediate corpus and pass results through ClariQ eval script. 

Each script is reliant on proper command line arguments to function.
Each required command line argument can be found near the top of the file as an argument passed to the `get_arg` function.
When a ColBERT run config and ColBERT config is required, additional arguments must be passed that can be found in the 
file `utils/__init__.py` under the `params` variable. 
Please contact me if anything is unclear.

## Datasets

Two datasets are required.

1. ClariQ - can be downloaded from https://github.com/aliannejadi/ClariQ.
   Must be place in project root directory.
2. WikiText - can be downloaded from https://huggingface.co/datasets/wikitext.
   File must be stored with structure
    - `root_wikitext_dataset_path/wikitext2/[test, train, valid].txt`
    - `root_wikitext_dataset_path/wikitext103/[test, train, valid].txt`
    Root wikitext dataset path should be specified with `dataset_path` command line argument for scripts that require it.
      
## Dependencies

ColBERT must be installed according to installation instructions found at https://github.com/stanford-futuredata/ColBERT/tree/new_api