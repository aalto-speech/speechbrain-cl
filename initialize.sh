#!/bin/bash

env_dir="sssb"

function err() {
  echo "$1";
  exit 1;
}

function create_env() {

  if [[ ! -f $env_dir/bin/activate ]] ; then
    read -p "Could not find the python-venv directory ${env_dir}. Create it? [y|N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] ; then
      python -m venv "$env_dir" || err "Could not create new venv. Are you sure the package exists?"
    else
      echo "$0: Aborting..."
      exit 1;
    fi
  fi

  source "$env_dir"/bin/activate || exit 1;

  # Install dependencies
  python -m pip install -r requirements.txt || exit 1;

  # Install speechbrain
  if [[ ! -f "./speechbrain/setup.py" ]] ; then
    # Make sure the submodule is up-to-date
    git submodule update --init --recursive
  fi
  cd speechbrain && python -m pip install -e . && cd ..|| exit 1;

  # Install current package (imported as src.curriculum)
  python -m pip install -e .

  exit 0;
}

# Try to load the python3.8 module (triton).
# If this fails (then we are not on triton probably) 
# and we will simply create a new env.
# Load python>3.6 (for speechbrain)
module load python/3.8.7 || create_env
echo "$0: Loaded python 3.8.7"

#module load cuda/10.2.89
#echo "$0: Loaded cuda/10.2.89"

# Create environment on triton
create_env

# Now assuming that you have a myrun.sh script you want to run, you can do:
#chmod +x ./myrun.sh
#TRITON_MEM=32G
#TRITON_TIME="4:30:00"
#TRITON_GPU="gpu:1"
#srun --mem="$TRITON_MEM" --time="$TRITON_TIME" --gres="$TRITON_GPU" ./myrun.sh

exit 0;
