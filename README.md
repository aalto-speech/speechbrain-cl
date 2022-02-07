# Curriculum Learning Methods for Speechbrain

## Installation

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
python -m pip install -e .
```

## Example Usage

Let's say you have your own repository named `myasr` and want to test this package. A simple way to do it would be to clone this repo inside your repository and then install it. An alternative would be to add this project as a submodule to your git repo which will make it easier for you to pull changes. Example (assuming you are in an already existing python venv):

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


## How to Use

1. Clone the project.
2. (Optional) Create a virtual env.
3. Install the package: `python -m pip install -e .` (in editable mode).
4. Create a `.yaml` hyperparameters file. Check `cl/hparams_examples` for an example. This is not very well documented yet.

## TODO:

- Fill readme with instructions.
- Document how `cl/hparams_examples/` work.
- Add tests.
