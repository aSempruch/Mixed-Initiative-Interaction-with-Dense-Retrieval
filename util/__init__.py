import os
import sys
import argparse
from colbert.infra import RunConfig, ColBERTConfig
from typing import Dict, Callable
import ast

global_required = ['experiment']


def get_arg(arg: str):
    all_args = argparse.ArgumentParser()
    all_args.add_argument(f'--{arg}', required=True)

    return list(vars(all_args.parse_known_args()[0]).values())[0]


def _config_from_args(
        config_obj: Callable,
        params: Dict,
        required = None
):

    global global_required
    all_required = [*global_required, *required] if required else global_required

    all_args = argparse.ArgumentParser()

    for param, default in params.items():
        arg_type = type(default)
        all_args.add_argument(
            f'--colbert_{param}',
            type=arg_type if arg_type != type(None) else str,
            required=(param in all_required)
        )

    args = vars(all_args.parse_known_args()[0])

    args_no_prefix = {arg.replace('colbert_', ''):val for (arg, val) in args.items()}

    return config_obj(**args_no_prefix)


def run_config_from_args(required = None) -> RunConfig:

    params = {
        'name': None,
        'experiment': 'default',
        'nranks': 1
    }

    return _config_from_args(RunConfig, params, required=required)


def colbert_config_from_args(required = None) -> ColBERTConfig:

    params = {
        'name': None,
        'doc_maxlen': 220,
        'dim': 128,
        'mask_punctuation': True,
        'similarity': 'cosine',
        'bsize': 32,
        'index_name': None
    }

    return _config_from_args(ColBERTConfig, params, required=required)
