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
from . import utils
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


strip_spaces = lambda s: utils.strip_spaces(s)

logger = logging.getLogger(__name__)

""" A wrapper around `DynamicItemDataset` which will change the way the dataset
    is sorted. In addition, it aims at filtering out the "hard" examples.
"""
class CurriculumDataset(DynamicItemDataset):
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
    
    # def curriculum_sort(self,
    #     brain: sb.core.Brain,
    #     current_epoch: int,
    #     progressbar: bool = True,
    #     keep_confs: bool = False,
    # ) -> FilteredSortedDynamicItemDataset:
    #     # NOTE: Assumes that `sorting` is a valid sorting method taken from
    #     #       the CURRICULUM_KEYS.
    #     # Before the first epoch sort by length since the losses don't mean anything.
    #     # We assume that this is checked from the ASR model and that's why we 
    #     # raise an error otherwise.
    #     if current_epoch == 1:
    #         raise Exception("The first epoch should be handled by the \
    #             asr model. See asr_model.py of the CommonVoice recipe")
    #     self.sorting = brain.sorting
    #     if (brain.sorting in self.LOSS_SORTERS) and \
    #       (current_epoch <= brain.hparams.number_of_ctc_epochs) \
    #           and ('ctc' in brain.sorting):
    #         logger.info(strip_spaces(f"Converting sorting method from: \
    #             {brain.sorting} to seq_loss since we surpassed the max \
    #                 number of ctc epochs."))
    #         self.sorting = 'seq_loss'
    #     # Make sure that the loss sorters and metric sorters complete 
    #     # the whole set of curriculum keys
    #     assert set(self.LOSS_SORTERS + self.METRIC_SORTERS) == set(self.CURRICULUM_KEYS)
    #     # Make sure that the sorting method is either a single method 
    #     # or a combination of no more than 2 methods
    #     if isinstance(self.sorting, tuple) or isinstance(self.sorting, list):
    #         # Make sure that the input sorting method(s) is a valid one
    #         assert set(self.sorting).issubset(self.CURRICULUM_KEYS), f"Invalid method \
    #             {self.sorting} (out of: {self.CURRICULUM_KEYS})."
    #         assert len(self.sorting) <= 2
    #     # Set evaluation mode
    #     brain.modules.eval()
    #     # Reset nonfinite count to 0 each epoch
    #     brain.nonfinite_count = 0
    #     # How to handle taking a subset.
    #     if hasattr(brain.hparams, 'iterative_trainset_increase') and brain.hparams.iterative_trainset_increase is True:
    #         n_initial_datapoints = brain.hparams.initial_trainset_perc * len(self)
    #         logger.debug(f">>> {n_initial_datapoints=}")
    #         # logger.info(f">>> {epochs_with_increase=}")
    #         # Number of epochs that we have been increasing the size.
    #         # We subtract 1 because the first epoch is always ascending sorting.
    #         n_increasing_epochs = current_epoch - 1
    #         # For how many epochs do we want to increase the size of our dataset?
    #         n_subset_epochs = getattr(
    #             brain.hparams, 'number_of_subset_epochs', 
    #             brain.hparams.number_of_curriculum_epochs-1
    #         )
    #         assert n_subset_epochs <= brain.hparams.number_of_curriculum_epochs
    #         # How much do we want to increase the length of the trainset in each epoch
    #         n_increasing_datapoints = (len(self) - n_initial_datapoints) / n_subset_epochs
    #         logger.debug(f">>> {n_increasing_datapoints=}")
    #         # How many datapoints should we use in the current epoch (for training)
    #         self.current_epoch_n_datapoints = n_initial_datapoints + \
    #             np.ceil((n_increasing_epochs * n_increasing_datapoints))
    #         logger.info(strip_spaces(f"When we get to the training stage, we will use \
    #             {round(self.current_epoch_n_datapoints)} datapoints which is \
    #             {round(100*self.current_epoch_n_datapoints/len(self), 2)}% of \
    #                 the whole training set."))
    #     else:
    #         # This will be handled by select_n in filter_sorted_ids
    #         self.current_epoch_n_datapoints = None
        
    #     logger.info(f"Sorting the trainset based on the {self.sorting} value...")
    #     enable = progressbar and sb.utils.distributed.if_main_process()
    #     # Default arguments for curriculum sorting
    #     kwargs={
    #         "brain": brain,
    #         "enable": enable,
    #         "keep_confs": keep_confs,
    #         "current_epoch": current_epoch,
    #         "running_average": self.running_average,
    #     }
    #     # Decide on the curriculum-sorting method
    #     if self.sorting in self.LOSS_SORTERS:
    #         # E.g. seq2seq loss sorting
    #         kwargs['curriculum_class'] = LossCurriculum
    #     elif self.sorting in self.METRIC_SORTERS:
    #         # I.e. wer/cer sorting
    #         kwargs['curriculum_class'] = MetricCurriculum
    #     else:
    #         # E.g. Combination of seq2seq and wer/cer sorting
    #         # Define the order of the sorting methods
    #         if self.sorting[0] in self.LOSS_SORTERS:
    #             kwargs['loss_method'], kwargs['metric_method'] = self.sorting
    #         else:
    #             kwargs['metric_method'], kwargs['loss_method'] = self.sorting
    #         kwargs['curriculum_class'] = JointCurriculum
    #     self._get_curriculum_dict(**kwargs)
    #     # post_kwargs['sorting_dict'] = sorting_dict
    #     logger.info(f"Number of datapoints: {len(self.sorting_dict)}")
    #     # import json
    #     # logger.info(f"Sorting Dict:{json.dumps(sorting_dict, indent=4)}")
    #     return self.sorting_dict
    
    # def _get_curriculum_dict(self, 
    #     brain: sb.core.Brain,
    #     curriculum_class: BaseCurriculum,
    #     keep_confs: bool = False,
    #     current_epoch: int = None,
    #     enable: bool = True,
    #     **kwargs
    # ):
    #     logger.info("Calculating the sorting dictionary...")
    #     current_epoch = current_epoch or brain.hparams.epoch_counter.current
    #     curriculum_object: BaseCurriculum = curriculum_class(
    #         sorting_method=self.sorting,
    #         brain=brain,
    #         sorting_dict=self.sorting_dict,  # by reference
    #         keep_confs=keep_confs,
    #         **kwargs
    #     )
    #     train_set: torch.utils.data.DataLoader = brain.make_dataloader(self, 
    #         stage=sb.Stage.TRAIN, ckpt_prefix="initial-curr-dataloader-",
    #         **brain.train_loader_kwargs
    #     )
    #     if len(self.sorting_dict) < (len(train_set)-1)*brain.hparams.batch_size:
    #         last_ckpt_time = time.time()
    #         starting_time = time.time()
    #         initial_step = self.step if brain.distributed_launch else 0
    #         with tqdm(
    #             train_set,
    #             initial=initial_step,
    #             total=len(train_set),
    #             dynamic_ncols=True,
    #             disable=not enable,
    #         ) as t:
    #             t.set_postfix_str(f"Curriculum Sorting - Epoch: {current_epoch}.")
    #             logger.info(f"Starting step: {self.step}, len: {len(train_set)}")
    #             batch_iterator = t if brain.distributed_launch else islice(t, self.step, None)
    #             for batch in batch_iterator:
    #                 self.step += 1
    #                 # will update the sorting dict and the running average
    #                 self.running_average = curriculum_object._process_batch(
    #                     batch, 
    #                     self.running_average, 
    #                     self.step
    #                 )
    #                 t.set_postfix(
    #                     running_average=self.running_average, 
    #                     epoch=current_epoch
    #                 )
    #                 if (
    #                     brain.checkpointer is not None
    #                     and brain.ckpt_interval_minutes > 0
    #                     and time.time() - last_ckpt_time >= brain.ckpt_interval_minutes * 60.0
    #                 ):
    #                     # run_on_main(curriculum_object.save_dict)
    #                     run_on_main(brain._save_intra_sorting_ckpt)
    #                     last_ckpt_time = time.time()
    #         # Make sure we have iterated all of the training samples
    #         assert len(self.sorting_dict) >= (len(train_set)-1)*brain.hparams.batch_size, f"{len(self.sorting_dict)=} ====== {len(train_set)=}"
    #         total_dur = (time.time() - starting_time)/60

    #         run_on_main(brain._on_curriculum_end, kwargs={
    #             "running_average": self.running_average,
    #             "time_to_sort": total_dur,
    #             "epoch": current_epoch,
    #         })
    #         logger.info(strip_spaces(f"Sorting by {self.sorting} complete. \
    #             Sorting Duration: {total_dur} minutes."))
    #     else:
    #         logger.info(f"Will use the loaded dictionary..")
    #     self.step = 0

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

    def _curriculum_filtered_ids(
        self,
        sorting_dict: Dict[str, Union[float, Tuple[float, float], Tuple[float, float, float]]],
        reverse: bool = False,
        select_n: Optional[int] = None,
        debug: bool = False,  # set to True to make some extra assertions
        epsilon: float = 1e-3,
    ) -> List[str]:
        select_n = select_n or (getattr(self, "current_epoch_n_datapoints", None) or len(sorting_dict))
        select_n = round(select_n)
        # logger.info(f"Will select {select_n} datapoints out of {len(sorting_dict)} in total.")
        sd0 = list(sorting_dict.values())[0]
        if isinstance(sd0, tuple) and self.sorting in self.METRIC_SORTERS:
            # Then we need to sort the dictionary based on a 
            # combination of the loss/metric value and the
            # utterance's duration and/or confidence.
            sorting_dict = utils.normalize_with_confs(sorting_dict, epsilon)
            # import json
            # logger.info(f"Sorting Dict:{json.dumps(sorting_dict, indent=4)}")
        elif self.sorting in self.METRIC_SORTERS:
            # In this case the value of the dictionary is a single float
            # Normalize to [0, 1] (min-max normalization)
            sorting_dict = utils.normalize_dict(sorting_dict)
        elif self.sorting in self.LOSS_SORTERS and isinstance(sd0, tuple) and len(sd0) == 2:
            sorting_dict = utils.normalize_only_durs(sorting_dict)
        # sort the ids (dict keys) based on their value and keep N
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