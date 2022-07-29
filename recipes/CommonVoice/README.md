# Common Voice recipe (version 8)

This repository contains the CL recipe for the english version of common voice version 8 (CV8). We assume that you have downloaded the data from the official source (mozilla common voice project) and that you have extracted the compressed version into a normal file (this might be tricky since there are many audio files in cv8).

The file `prepare_cv8.py` contains the code for creating the necessary train/dev/test csv files that speechbrain is going to process. This script is executed from inside the `train.py` script which is responsible for training and validating your model. To make sure that everything runs smoothly, check that the hyperparameters (and most importantly the data paths) are set correctly in `./ASR/seq2seq/hparams/crdnn_wer_noisy.yaml`. **In all `.yaml` files, the `data_folder` is a placeholder**. This means that when calling the `train.py` script you should also pass the argument `--data_folder="<path-to-cv8>"`.

## Example Usage:

From inside `ASR/seq2seq/` run the following:

```
python train.py hparams/crdnn_wer_noisy.yaml --device=cuda:0 --data_folder=/mnt/data/common_voice8/en/
```