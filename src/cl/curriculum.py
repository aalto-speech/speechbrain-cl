import copy
import logging
import time
import random
import os
from itertools import islice
from typing import (
    List, Tuple, Callable, Optional, Dict,
    Union, Sequence
)
# from collections.abc import Iterable
import numpy as np
from tqdm import tqdm
from cl.utils.process_utils import (
    strip_spaces, normalize_dict, 
    normalize_only_durs, normalize_with_confs
)
# from .classes import (
#     MetricCurriculum, LossCurriculum, 
#     JointCurriculum, BaseCurriculum
# )
import torch
import speechbrain as sb
from speechbrain.dataio.dataset import (
    DynamicItemDataset, 
    FilteredSortedDynamicItemDataset
)
from speechbrain.utils.distributed import run_on_main
# from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class CurriculumBase(DynamicItemDataset):
    def split_into_k(self, 
      k: int, 
      reverse: Optional[bool] = False, 
      sorting_dict: Optional[dict] = None,
      incremental: Optional[bool] = False,
    ) -> List[np.ndarray]:
        """
        Arguments:
            k: Number of difficulty groups. E.g. if `reverse` is False then the first
               group will contain the easiest examples and the last one the hardest ones.
            reverse: If true then the subsets will be returned by order "hardest to easiest".
            sorting_dict: The dictionary containing utterance ids as keys and scores as values.
            incremental: If true then each consecutive sub-array will also contain the previous 
              samples.
        Returns:
            A list of `k` numpy arrays of equal length. If incremental is True then
            each array A_i will contain A_{i-1} + A_i.
        """
        sorting_dict = sorting_dict or {}
        if len(self.sorting_dict) == 0 and len(sorting_dict) == 0:
            raise ValueError("The class' dictionary is empty, so you need to pass a valid `sorting_dict` argument.")
        sorting_dict = sorting_dict or self.sorting_dict
        sorted_ids = sorted(sorting_dict, key=lambda x: sorting_dict[x], reverse=reverse)
        splitted = np.array_split(sorted_ids, k)
        if not incremental:
            return splitted
        out = [None]*len(splitted)
        out[0] = splitted[0]
        for i, arr in enumerate(splitted[1:]):
            out[i+1] = np.concatenate((out[i], arr), axis=0)
        return out

    def adaptive_pacing(self,
        sorting_dict: dict,
        n_difficulty_groups: int,
        epochs_per_group: int,
        incremental: bool = True,
        noise_percentage: Optional[float] = None,
        normalize: Optional[bool] = True,
        reverse: Optional[bool] = False,
        current_epoch: Optional[int] = 0,
    ):
        """
        Arguments:
            sorting_dict: The sorting dictionary (scores of each utterance).
            n_difficulty_groups: Number of difficulty groups. Check CurriculumDataset.split_into_k
                for more information.
            epochs_per_group: On how many epochs should each group be used for training?
                E.g. if 2, then the easiest group will be used for 2 epochs, then the
                     next group will be used for the next 2 epochs, and so on.
            incremental: If true then each subsequent subset will also contain the easy 
                examples taken from the previous subset. Check CurriculumDataset.split_into_k for more.
            noise_percentage: For noisy CL. Check CurriculumDataset.filtered_sorted_ids and 
                self.add_random_noise for more.
            normalize: Whether or not the sorting dictionary should be normalized. Notice that
                this normalization is IN-PLACE if inplace is True. The same normalization happens in
                CurriculumDataset._curriculum_filtered_ids
            reverse: Descending sorting?
        """
        logger.info(f"Number of difficulty groups (k): {n_difficulty_groups=}, {epochs_per_group=}")
        if not isinstance(sorting_dict, dict) or len(sorting_dict) == 0:
            raise ValueError(f"Invalid sorting dictionary of type: {type(sorting_dict)}.")
        if normalize:
            sorting_dict = self.normalize_dict(sorting_dict)
        paced_sorted_ids = self.split_into_k(
            k=n_difficulty_groups,
            reverse=reverse,
            sorting_dict=sorting_dict,
            incremental=incremental
        )
        tmp_path = "/m/teamwork/t40511_asr/p/curriculum-e2e/startover/test_recipes/lahjoita_puhetta/ASR/seq2seq/exps/tests/"
        with open(os.path.join(tmp_path, "paced_sorted_ids.txt"), "w") as fw:
            for i, el in enumerate(paced_sorted_ids):
                fw.write(f"{i=}:\t {len(el)=} \t[{', '.join(el[:10])}]\n\n\n\n\n")
        logger.info(f'Saved paced indices under {os.path.join(tmp_path, "paced_sorted_ids.txt")}')
        # self.adaptive_pacing_index is a tuple (in the form of a numpy array)
        # whose first element is the index of paced_sorted_ids which we will use,
        # and the second element is the number of epoch that this index has been used.
        # If the second element is greater than epochs_per_group then we move on to the
        # next group.
        logger.info(f"Adaptive pacing index before update: {getattr(self, 'adaptive_pacing_index', None)}")
        if not hasattr(self, "adaptive_pacing_index"):
            paced_ids_index = max(0, current_epoch // epochs_per_group - 1)
            n_usage_epochs = current_epoch % epochs_per_group - 1
            self.adaptive_pacing_index = np.array((paced_ids_index, n_usage_epochs))
        elif self.adaptive_pacing_index[0] >= len(paced_sorted_ids)-1:
            logger.warning(strip_spaces(f"The adaptive pacing index has reached the maximum number \
                of groups ({self.adaptive_pacing_index}). We will keep increasing the \
                number of epochs that this group has been used, though. Is this intentional?"))
        current_indices = paced_sorted_ids[self.adaptive_pacing_index[0]]
        logger.info(f"Number of training samples in the current group: {len(current_indices)}")
        # Increase the number of epochs this group has been used for.
        self.adaptive_pacing_index[1] += 1
        # If the number of epochs exceeds the `epochs_per_group` then
        # we move to the next group.
        if self.adaptive_pacing_index[1] >= epochs_per_group and self.adaptive_pacing_index[0] < len(paced_sorted_ids)-1:
            self.adaptive_pacing_index[0] += 1
            self.adaptive_pacing_index[1] = 0
        self.adaptive_pacing_index[0] = min(self.adaptive_pacing_index[0], len(paced_sorted_ids)-1)
        if isinstance(noise_percentage, float) and 0.0 < noise_percentage <= 1.0:
            current_indices = self.add_random_noise(current_indices, noise_percentage)
            logger.info("Added some random noise among the easy examples.")
        logger.info(f"Adaptive pacing index is: {self.adaptive_pacing_index}")
        return FilteredSortedDynamicItemDataset(self, current_indices)
    
    @classmethod
    def add_random_noise(cls, id_list: List[str], noise_percentage: float = 0.15):
        assert 0.0 < noise_percentage < 1
        # Step 1: Split list in 3 parts: [easy examples] [medium examples] [hard examples]
        n_ids = len(id_list)
        _n = n_ids // 3
        easy_ids = id_list[:_n]
        medium_ids = id_list[_n:_n*2]
        hard_ids = id_list[_n*2:]
        n_noisy_samples = int(noise_percentage*len(easy_ids))
        # Step 2: 60% of the noise will come from the hard samples and 40% from the medium ones
        n_samples_hard = round(0.6*n_noisy_samples)
        n_samples_med = round(0.4*n_noisy_samples)
        n_samples = n_samples_med + n_samples_hard  # avoid rounding errors, redo sum
        # Step 3: Sample some random ids.
        hard_samples = random.sample(hard_ids, n_samples_hard)
        # take non-common elements, in other words remove the sampled elements from the 
        # list of hard_ids since they will be moved to the "easy" list
        hard_ids = list(set(hard_ids) ^ set(hard_samples))
        medium_samples = random.sample(medium_ids, n_samples_med)
        # Similarly as with the hard ids
        medium_ids = list(set(medium_ids) ^ set(medium_samples))
        # Step 4: Sample an equivalent number of ids from the easy samples.
        #         These ids are the ones that are going to be replaced.
        easy_sample_ids = random.sample(range(len(easy_ids)), n_samples)
        for sample_index in easy_sample_ids:
            if len(hard_samples) > 0:
                # First add all hard samples and then move to the medium ones
                new_val = hard_samples.pop()
                list_to_append = hard_ids  # just a reference
            else:
                new_val = medium_samples.pop()
                list_to_append = medium_ids
            old_val = easy_ids[sample_index]
            easy_ids[sample_index] = new_val
            list_to_append.append(old_val)
        out = easy_ids + medium_ids + hard_ids
        # logger.info(f"Initial id list: {id_list[:20]}\nFinal id list: {out[:20]}")
        assert len(out) == len(id_list), f"{len(out)=} != {len(id_list)=}\n{out=}\n{id_list=}"
        return out

""" A wrapper around `DynamicItemDataset` which will change the way the dataset
    is sorted. In addition, it aims at filtering out the "hard" examples.
"""
class CurriculumDataset(CurriculumBase):
    # TODO: Add the curriculum specific method-names here.
    CURRICULUM_KEYS = ['ctc_loss', 'seq_loss', 'seq_ctc_loss', 'wer', 'cer']
    LOSS_SORTERS = ['ctc_loss', 'seq_loss', 'seq_ctc_loss']
    METRIC_SORTERS = ['wer', 'cer']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
        self.sorting_dict = {}
        self.running_average = 0.0
        self.sorting = kwargs.get('sorting', None)
    
    def _add_duration_key(self):
        # No need to use 'duration' when we have the default implementation.
        if self._sorting in self.CURRICULUM_KEYS:
            # Add the audio's duration to the output keys (utterance length)
            original_keys = self.pipeline.output_mapping
            original_keys['duration'] = 'duration'
            original_keys['id'] = 'id'
            sb.dataio.dataset.set_output_keys([self], original_keys,)
    
    @property
    def sorting(self):
        if not hasattr(self, "_sorting"):
            return None
        return self._sorting
    
    @sorting.setter
    def sorting(self, value):
        self._sorting = value
        # If we are on a curriculum method then add the duration key.
        self._add_duration_key()
    
    @sorting.deleter
    def sorting(self):
        if self._sorting in self.CURRICULUM_KEYS:
            # Remove 'duration' if it existed
            original_keys = self.pipeline.output_mapping
            original_keys.pop("duration", None)
            sb.dataio.dataset.set_output_keys([self], original_keys,)
        del self._sorting

    # This function will be called before each
    def filtered_sorted(self,
        key_min_value: Optional[dict] = {},
        key_max_value: Optional[dict] ={},
        key_test: Optional[dict] = {},
        sort_key: Optional[str] = None,
        reverse: bool = False,
        select_n: int = None,
        sorting_dict: Optional[dict] = None,
        hparams: Optional[dict] = None,
        noise_percentage: Optional[float] = None,
    ) -> FilteredSortedDynamicItemDataset:
        if sort_key not in (self.CURRICULUM_KEYS + ['random']) or (not sorting_dict):
            # If the function is not called for "curriculum learning" 
            # then use the default behavior
            filtered_sorted_ids = self._filtered_sorted_ids(
                key_min_value, key_max_value, key_test, 
                sort_key, reverse, select_n,
            ) 
        elif (sorting_dict is None) or not isinstance(sorting_dict, dict):
            # If the function is expected to sort the training set based on a
            # curriculum method, then it should provide a 'sorting_dict' dictionary.
            # The dictionary should contain data point ids as keys and the
            # corresponding loss on the current epoch.
            raise ValueError("You provided a curriculum sorting key but no losses. Aborting...")
        else:
            # Normal behavior for curriculum learning.
            sorting_dict.pop('num_datapoints', None)
            filtered_sorted_ids: list = self._curriculum_filtered_ids(
                sorting_dict, reverse, select_n,
            )
        if isinstance(noise_percentage, float) and 0.0 < noise_percentage <= 1.0:
            # logger.info(f"{filtered_sorted_ids[:10]=}")
            filtered_sorted_ids = CurriculumDataset.add_random_noise(filtered_sorted_ids, noise_percentage)
            logger.info("Added some random noise among the easy examples.")
            # logger.info(f"{filtered_sorted_ids[:10]=}")
        filtered_trainset = FilteredSortedDynamicItemDataset(self, filtered_sorted_ids)
        return filtered_trainset

    def normalize_dict(self, sorting_dict, select_n: Optional[int]=None, epsilon: float = 1e-3,):
        select_n = select_n or (getattr(self, "current_epoch_n_datapoints", None) or len(sorting_dict))
        select_n = round(select_n)
        # logger.info(f"Will select {select_n} datapoints out of {len(sorting_dict)} in total.")
        sd0 = list(sorting_dict.values())[0]
        if isinstance(sd0, tuple) and self.sorting in self.METRIC_SORTERS:
            # Then we need to sort the dictionary based on a 
            # combination of the loss/metric value and the
            # utterance's duration and/or confidence.
            sorting_dict = normalize_with_confs(sorting_dict, epsilon)
            # import json
            # logger.info(f"Sorting Dict:{json.dumps(sorting_dict, indent=4)}")
        elif self.sorting in self.METRIC_SORTERS:
            # In this case the value of the dictionary is a single float
            # Normalize to [0, 1] (min-max normalization)
            sorting_dict = normalize_dict(sorting_dict)
        elif self.sorting in self.LOSS_SORTERS and isinstance(sd0, tuple) and len(sd0) == 2:
            sorting_dict = normalize_only_durs(sorting_dict)
        return sorting_dict

    def _curriculum_filtered_ids(
        self,
        sorting_dict: Dict[str, Union[float, Tuple[float, float], Tuple[float, float, float]]],
        reverse: bool = False,
        select_n: Optional[int] = None,
        debug: bool = False,  # set to True to make some extra assertions
        epsilon: float = 1e-3,
    ) -> List[str]:
        sorting_dict = self.normalize_dict(sorting_dict, select_n=select_n, epsilon=epsilon)
        filtered_sorted_ids: list = [i for i in sorted(
            sorting_dict,
            key=lambda x: sorting_dict[x],
            reverse=reverse)
        ][:select_n]
        if debug:
            # Make sure that the sorting was successfull (debugging).
            for i, j in zip(filtered_sorted_ids[:-1], filtered_sorted_ids[1:]):
                if reverse:
                    assert sorting_dict[i]>=sorting_dict[j], f"i:{i}, j:{j}, di: {sorting_dict[i]}, dj: {sorting_dict[j]}" 
                else:
                    assert sorting_dict[i]<=sorting_dict[j], f"i:{i}, j:{j}, di: {sorting_dict[i]}, dj: {sorting_dict[j]}" 
            # Make sure that we have the valid batch ids
            for i, data_id in enumerate(self.data_ids):
                assert data_id in sorting_dict.keys(), f"Could not locate {data_id}."
        return filtered_sorted_ids

    def _on_curriculum_end(self, brain: sb.core.Brain, original_keys: dict, sorting_dict: dict = None):
        # Used to save a checkpoint. Could be useful for multi-gpu
        pass
        
    @classmethod
    def _save_ordering(cls, filtered_sorted_ids: list, model_folder: str, epoch: int):
        out_folder = os.path.join(model_folder, 'curriculums')
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f'epoch-{epoch}.txt')
        with open(out_path, 'w') as fw:
            fw.write('\n'.join(filtered_sorted_ids))
        logger.info("Saved the CL ordering under {}.".format(out_path))


# TODO: This is more or less the same as FilteredSortedDynamicItemDataset. Converge the two.
class CurriculumSubset(CurriculumDataset):
    def __init__(self, dataset: CurriculumDataset, indices: Sequence[int], *args, **kwargs) -> None:
        self.dataset = dataset
        self.data = dataset.data
        self.indices = indices
        self.data_ids = [data_id for idx, data_id in enumerate(self.data.keys()) if idx in indices]
        # super().__init__(data=dataset.data, *args, **kwargs)
        self.pipeline = copy.deepcopy(dataset.pipeline)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)