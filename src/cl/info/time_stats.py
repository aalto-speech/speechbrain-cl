import os, glob
from typing import Optional
from tqdm import tqdm
import numpy as np
from cl.curriculum import CurriculumDataset
from cl.utils.process_utils import calculate_dataset_hours

def calculate_total_hours_seen(
        train_csv: str, 
        n_epochs: int, 
        is_paced: bool = False,
        subsampling_n_epochs: Optional[int] = None,
        is_fixed: bool = False,
    ):
    assert os.path.isfile(train_csv), train_csv
    assert isinstance(n_epochs, int) and n_epochs > 0, "n_epochs must be a positive integer."
    total_hours_seen = 0.
    dataset = CurriculumDataset.from_csv(train_csv)
    if is_paced:
        assert isinstance(subsampling_n_epochs, int), "subsampling_n_epochs should define \
            every how many epochs the pacing function is applied"
        curr_logs_path = os.path.join(os.path.dirname(train_csv), 'curriculum_logs')
        assert os.path.isdir(curr_logs_path), curr_logs_path
        shuffled_ids_paths = glob.glob(os.path.join(curr_logs_path, "*.npy"))
        epochs_seen = 1
        for i, f in tqdm(enumerate(shuffled_ids_paths)):
            relevant_ids = np.load(f)
            # print(f"{type(dataset)=}\n{type(dataset.data)=}\n{relevant_ids.shape}")
            data = [dataset.data[data_id]['duration'] for idx, data_id in enumerate(dataset.data.keys()) if idx in relevant_ids]
            hours = (sum(data)/60)/60
            # The pacing function is applied every subsampling_n_epochs epochs
            # so we need to take that into account.
            if i == 0:
                hours *= (subsampling_n_epochs-1)
                epochs_seen += subsampling_n_epochs-1
            elif i == len(shuffled_ids_paths)-1:
                # At the last epoch we get the remaining hours
                hours *= max(1, n_epochs-epochs_seen)
                pass
            else:
                epochs_seen += min(subsampling_n_epochs, max(1, n_epochs-((i+1)*subsampling_n_epochs)))
                hours *= min(subsampling_n_epochs, max(1, n_epochs-((i+1)*subsampling_n_epochs)))
            total_hours_seen += hours
        print("Saw {} epochs that saw {} hours of data.".format(epochs_seen, total_hours_seen))
    else:
        total_hours_seen = calculate_dataset_hours(dataset) * n_epochs
    return total_hours_seen