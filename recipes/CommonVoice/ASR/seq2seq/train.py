#!/usr/bin/env python3
""" Recipe for training a sequence-to-sequence ASR system on 
the Common Voice v8 dataset.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch.

To run this recipe, do the following:
> python train.py hparams/<my-hyperparameters>.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard GRU and BeamSearch (no LM).

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE) 
and many other possible variations.

Authors
 * Georgios Karakasidis 2022
"""

import sys
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from cl.train_utlis import fit
from cl.asr_models import ASR, AsrWav2Vec2


logger = logging.getLogger(__name__)

def main():
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation
    from prepare_cv8 import prepare_cv8  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )        

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_cv8,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
            "min_duration_secs": hparams["min_duration_secs"],
            "max_duration_secs": hparams["max_duration_secs"],
            "train_dir": hparams.get("train_dir", "train-100h"),
            "dev_dir": hparams.get("dev_dir", "dev"),
            "test_dir": hparams.get("test_dir", "test"),
            "train_csv": hparams.get("train_subset_path", None),
        },
    )
    ASR_Model = ASR
    if hparams.get('use_wav2vec2', False):
        ASR_Model = AsrWav2Vec2
    fit(
        hparams, run_opts, overrides, ASR_Model=ASR_Model
    )

if __name__ == "__main__":
    main()