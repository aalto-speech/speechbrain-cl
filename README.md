# Curriculum Learning Methods for Speechbrain

[![PyPI](https://img.shields.io/pypi/v/speechbrain-cl.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/speechbrain-cl.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/speechbrain-cl)][python version]
[![License](https://img.shields.io/pypi/l/speechbrain-cl)][license]

[![Read the documentation at https://speechbrain-cl.readthedocs.io/](https://img.shields.io/readthedocs/speechbrain-cl/latest.svg?label=Read%20the%20Docs)][read the docs]
<!-- [![Tests](https://github.com/geoph9/speechbrain-cl/workflows/Tests/badge.svg)][tests] -->
<!-- [![Codecov](https://codecov.io/gh/geoph9/speechbrain-cl/branch/main/graph/badge.svg)][codecov] -->
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/speechbrain-cl/
[status]: https://pypi.org/project/speechbrain-cl/
[python version]: https://pypi.org/project/speechbrain-cl
[read the docs]: https://speechbrain-cl.readthedocs.io/
<!-- [tests]: https://github.com/geoph9/speechbrain-cl/actions?workflow=Tests -->
<!-- [codecov]: https://app.codecov.io/gh/geoph9/speechbrain-cl -->
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

- Implementation of multiple curriculum learning methods.
- Scoring functions: shown to improve performance (on the expense of training time).
- Pacing functions: improve training time (performance is on par to speechbrain's baseline).
- Works as a library on top of `speechbrain`.


## Installation

You can install _Speechbrain Cl_ via [pip] from [PyPI]:

```console
$ pip install speechbrain-cl
```

If you are using poetry, then do `poetry add speechbrain-cl`.

**Important Note:** This is intended to work on linux systems and you will probably encounter issues if you try to run this package on windows (and OS-X, too, probably).

### From Github

*Has only been tested on linux systems.*

```bash
# Clone repository
git clone git@github.com:aalto-speech/speechbrain-cl.git
# Create a virtual environment
python -m venv env
# Source environment
source ./env/bin/activate
```

- **Option 1:** Non-editable mode
```bash
# Install `build` and build the package
python -m pip install build && python -m build
# Install the created .whl file
python -m pip install dist/cl-1.0a0-py2.py3-none-any.whl
```

- **Option 2:** Editable mode (you can perform changes to the source code):
```bash
# Install package in editable mode
# You first need to install `wheel` to avoid bdist_wheel errors.
python -m pip install wheel
python -m pip install -e .
```

## Usage

Please see the [Command-line Reference] for details.

The installed package is called `cl` and it can be used both as library and a CLI tool. The CLI is primarily intended for analyzing results of already trained models and may be buggy. Of course, they could still be useful and that's why the CLI options are documented. To check them, run `cl --help`.

As a library, `cl` offers two main interfaces:
    - `cl.train_utils.fit`: Use this to fit a model based on some hyperparameters.
    - `cl.asr_models.ASR`: The default ASR model implementation. Assumes a CRDNN encoder and an RNN with attention as the decoder. If you want to use a different architecture then it is advised to base you code on this class.

Check `examples/train.py` for an example of how to utilize them.

The CL strategies are defined in the `cl.curriculum` file, while the `cl.base_asr_model.BaseASR` is the base class that handles updating the data loader (each time the CL strategy updates the ordering) and logging the necessary information.

### Command Line Interface

### More Advanced Usage

Let's say you have your own repository named `myasr` and want to test this package while also having the ability to tweak it. A simple way to do it would be to clone this repo inside your repository and then install it. An alternative would be to add this project as a submodule to your git repo which will make it easier for you to pull changes. Example (assuming you are in an already existing python venv):

```bash
# Add a submodule with the name `cl`
git submodule add git@github.com:aalto-speech/speechbrain-cl.git cl
# (Optional) You will probably also have to pull from the submodule:
git submodule update --init --recursive
# Install submodule
python -m pip install -e cl
# You can now either import the package or use its CLI:
cl --help  # print help message
```

Now if you want to update the package:

```bash
git submodule update --remote --merge
# Check if everything went fine and that the installation still works
python -m pip install -e cl
```

## Available CL methods

- **Metric-based**: The training set is sorted based on the results of a metric (e.g. WER or CER). By default we use the same model that we are trying to train in order to extract these values.
- **Loss-based**: Similar to the above, but instead uses seq2seq or seq2seq+ctc loss.
- **Random**: The training set is randomly shuffled at the beginning of the training. If you stop and restart training, the same (shuffled) training set will be loaded (contrary to the default behavior of speechbrain which would shuffle every time we create a dataloader).
- **Duration-based**: The default of speechbrain. Either ascending or descending. Leads to good results and also helps avoid using extra padding.
- **No sorting**: The training set will be processed in the same order as the data appear in the input file(e.g. `train.csv` file).

### CL sub-methods:

- **Transfer CL**: The training set is sorted based on an already trained model, but the sortings may change, depending on the performance on the currently training model. The new sortings will be taken by using one of the basic CL methods discussed above.
- **Transfer-fixed CL**: Same as above, except that the sortings never change (this is advised).
- **Subsample CL**: Every `N` epochs (`N` is a hyperparameter) a provided percentage of the training set is sampled and used for training for the next `N` epochs. The order of the training set is determined by using either transfer CL or any of the standard methods.
- **Subsample-incremental CL**: Same as above, but every `N` epochs we also increase the percentage of the training set that we are going to use for training.
- **Noisy CL**: Can we used with any of the above methods (except *No sorting* and *Duration-based*). It separates the training set into three categories: easy, medium-level and hard examples (the distinction happens by usign the sortings provided from the methods above). It then procceeds to add some hard and medium-level examples among the easy ones. This has helped with overfitting issues.

NOTE: The *Subsample* methods refer to curriculum learning with a combination of a scoring and a pacing function. The latter controls the flow with which the model sees the available training data. Check the subsampling area of the example recipe for more details.


## How to Use

1. Clone the project.
2. (Optional) Create a virtual env.
3. Install the package: `python -m pip install -e .` (in editable mode).
4. Create a `.yaml` hyperparameters file. Check `cl/examples/hyperparameters.yaml` for an example. This is not very well documented yet.
5. Copy the `train.py` file under `cl/examples` and adjust it to your needs. TODO: Maybe add the training option as part of the CLI.

### Adjustment of the hyperparameters

Choosing a CL method:

- `wer`: metric-based scoring function that uses the word error rate to rank the training samples. This means that a decoding pass is required at each training epoch.
- `cer`: similar to `wer` but uses character error rate instead.
- `seq_loss`: loss-based scoring function using the sequence-to-sequence loss to rank the training samples.
- `seq_ctc_loss`: Also uses the CTC loss values if CTC training is enabled.
- `ctc_loss`: CTC only loss-based scoring function.
- `random`: randomly shuffles the training data at the start of the first epoch. The same ranking will be kept even if you stop and resume training.
- `no`: the training set is read as it is and no re-ordering occurs. This can be seen as a special case of the `random` method.
- `ascending` and `descending`: duration-based CL.
- `char`/`word`/`token`: The ordering of the training samples occurs before the first epoch (just like with duration-based cl), but the criterion for the score of each utterance is the amount of rare characters/words/tokens in their content.

## Future Work:

- More CL methods (especially based on pacing functions).
- Add the training option as part of the CLI (the `prepare_dataset` function should be taken as an argument).
- Add tests.


## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Speechbrain Cl_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

Some well known issues:
- For `numpy` to work you need to have BLAS/LAPACK (and probably fortran, too). To install them on Ubuntu, run: `sudo apt-get -y install liblapack-dev libblas-dev gfortran`.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/geoph9/speechbrain-cl/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/geoph9/speechbrain-cl/blob/main/LICENSE
[contributor guide]: https://github.com/geoph9/speechbrain-cl/blob/main/CONTRIBUTING.md
[command-line reference]: https://speechbrain-cl.readthedocs.io/en/latest/usage.html