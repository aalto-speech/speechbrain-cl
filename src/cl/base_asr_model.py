#!/usr/bin/env python3
import ast
from copy import deepcopy
import os
import random
import re
import sys
import time
import numpy as np
from numpy.random import shuffle
from speechbrain.dataio.dataloader import LoopedLoader, SaveableDataLoader
from torch.utils.data.dataloader import DataLoader
import yaml
import torch
import logging
from tqdm import tqdm
import speechbrain as sb
from typing import Any, Dict, List, NamedTuple, Optional
from abc import ABC, abstractmethod
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.metric_stats import wer_details_for_batch
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.batch import PaddedBatch
# from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataset import DynamicItemDataset, FilteredSortedDynamicItemDataset
from cl.curriculum import CurriculumBase, CurriculumDataset, CurriculumSubset
from cl.methods.frequency_cl import FilteredSortedFrequencyCL, FrequencyCL
from cl.utils.process_utils import (
    checkpoint_wrapper_cl, strip_spaces, 
    load_sorting_dictionary, save_sorting_dictionary
)
# from speechbrain.utils.metric_stats import wer_details_for_batch


INTRA_SORTING_CKPT_FLAG = 'brain_intra_sorting_ckpt'
logger = logging.getLogger(__name__)

class YamlTupleSafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

YamlTupleSafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    YamlTupleSafeLoader.construct_python_tuple)

# Define training procedure
@sb.utils.checkpoints.register_checkpoint_hooks
class BaseASR(sb.core.Brain, ABC):

    DURATION_CL_KEYS = ['ascending', 'descending']
    VALID_CL_KEYS = ['ascending', 'descending', 'random']  + \
        CurriculumDataset.CURRICULUM_KEYS + FrequencyCL.VALID_FREQUENCY_TYPES
    CURRICULUM_KEYS = CurriculumDataset.CURRICULUM_KEYS

    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, 
                 checkpointer=None, sorting=None, train_set=None, train_loader_kwargs=None,
                 sorting_dict=None, final_sorting_dict=None):
        super(BaseASR, self).__init__(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams, 
            run_opts=run_opts, 
            checkpointer=checkpointer
        )
        self.sorting: str = (sorting or "").lower()
        self.train_set: CurriculumDataset = train_set
        if self.train_set is None:
            raise AttributeError("`train_set` argument not passed. You should have passed a valid speechbrain dataset instead.")
        self._curriculum_log_saved = False  # dummy flag to make sure we don't overwrite the logs
        self.train_set.sorting = self.sorting
        self.train_loader_kwargs: dict = train_loader_kwargs or {}
        # Epoch timing information
        self.start_epoch_time = time.perf_counter()
        self.current_training_time = 0.
        # if Curriculum Learning is chosen, for how many epochs should we use it?
        if not hasattr(self.hparams, 'number_of_curriculum_epochs'):
            self.hparams.number_of_curriculum_epochs = self.hparams.number_of_epochs
        # Whether to do profiling or not (helps in debugging)
        self.profiler = getattr(self.hparams, "profiler", False)
        # A dictionary keeping the "curriculum values" (orderings based on CL).
        # If we are using CL then it will also be saved and recovered during checkpointing.
        self.sorting_dict: dict = sorting_dict or final_sorting_dict or {}  # initialize it and it will be filled while training
        # This dictionary will only save the final sortings and not intermediate ones.
        # Only used for CL.
        # This is useful only when we recover a checkpoint and want to continue training
        #  from where we had left off.
        # E.g. We load a checkpoint on the 3rd epoch while being on batch number 40.
        #      So, on on_stage_start we will create a new dataloader (from the \
        #      complete sorting dict, a.k.a. the `final_sortings`). This way we are not
        #      mixing the current epoch's changes with the previous ones (the dataloader
        #      must be sorted based on the previous values of the dictionary).
        # TODO: In some cases this is a raw copy of self.sorting_dict which means we 
        #       keep the same dict twice in memory. Check when we can avoid this.
        self.final_sortings: dict = final_sorting_dict
        if self.final_sortings is None:
            self.final_sortings = (self.sorting_dict or {}).copy()
        # If we are loading a checkpoint on on_fit_start then we are going to 
        #  overwrite the dataloader on on_stage_start (due to curriculum).
        self._loaded_checkpoint = False  # used in utils.py (checkpoint_wrapper_cl function)
        # Use VAD in test set decoding?
        # If yes then this will probably result in better results in a non-segmented test set.
        # E.g. if you train a Brain model on 10second segments and the test set has around 
        #      1 minute long utterances, then using VAD will help you split in semgents and 
        #      get a better performance.
        self.VAD = getattr(self.hparams, 'VAD') if getattr(self.hparams, 'use_vad', False) is True else None
        # Whether to use fixed sortings or not (usefull in transfer learning CL approach)
        self.use_fixed_sorting = (getattr(self.hparams, "use_fixed_sorting", False) is True) and hasattr(self.hparams, "pretrained_model_hparams")
        # Prefix for dictionary save location
        self.dict_log_prefix = "subsample_" if self.do_subsample else ""
        # logger.info(f"Initialized brain. Type of train_set: {type(self.train_set)}.")
        self.sorting_dict_needs_save = (self.sorting != "random") and (not self.use_fixed_sorting) or self.do_subsample
        if self.use_fixed_sorting and len(self.final_sortings) == 0:
            try:
                self.final_sortings = self.load_sorting_dict(epoch=0, inplace=False)
                logger.info(f"Loaded dictionary with final sortings. Length: {len(self.final_sortings)}.")
            except AssertionError:
                if self.current_epoch > 0:
                    # Only warn after the first epoch
                    logger.warning(strip_spaces(f"{'='*80}\nCould not find a sorting dictionary even though you are using\
                        fixed transfer cl. This could lead to issues.\n{'='*80}"))

    @property
    def do_subsample(self):
        return getattr(self.hparams, 'do_subsample', False)

    @property
    def do_adaptive_pacing(self):
        return getattr(self.hparams, 'do_adaptive_pacing', False)
    
    @property
    def reverse(self):
        return getattr(self.hparams, "reverse", False)
    
    @property
    def epochs_per_adaptive_group(self):
        return getattr(self.hparams, "epochs_per_adaptive_group", 2) or 2
    
    @property
    def incremental_adaptive_pacing(self):
        return getattr(self.hparams, "incremental_adaptive_pacing", True)
    
    @property
    def curriculum_update_every(self):
        return getattr(self.hparams, 'curriculum_update_every', 1) or 1
    
    @property
    def current_epoch(self):
        return self.hparams.epoch_counter.current
    
    @current_epoch.setter
    def current_epoch(self, value):
        self.hparams.epoch_counter.current = value

    @property
    def current_trainset(self):
        if self.do_subsample and hasattr(self, 'train_subset'):
            return self.train_subset
        return self.train_set
    
    @property
    def use_transfer_cl(self):
        return self.use_fixed_sorting or (getattr(self.hparams, 'pretrained_model_hparams', None) is not None)

    @property
    def use_default_training(self):
        return (self.current_epoch <= getattr(self.hparams, 'default_sorting_epochs', 1)) and \
            (self.sorting in CurriculumDataset.CURRICULUM_KEYS)
    
    # Abstract methods MUST be implemented by the user.
    @abstractmethod
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        return

    @abstractmethod
    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        return
    
    @abstractmethod
    def on_valid_test_stage_start(self, stage):
        # The on_stage_start method for the VALID and TEST stages.
        # This needs to be different because on the TRAIN stage we also use curriculum.
        # An example implementation is to initialize the cer, wer computers.
        return

    def make_random_dataloader(self):
        assert self.sorting == "random", self.sorting
        assert self.hparams.do_subsample is True
        self.hparams.dataloader_options["shuffle"] = self.train_loader_kwargs["shuffle"] = False
        random_ids_path = os.path.join(
            self.hparams.output_folder,
            f"random_ids.npy"
        )
        if os.path.isfile(random_ids_path):
            shuffled_train_ids = np.load(random_ids_path)
        else:
            shuffled_train_ids = np.random.permutation(len(self.train_set))
            np.save(random_ids_path, shuffled_train_ids)
        self.train_set = CurriculumSubset(self.train_set, shuffled_train_ids)
        dataloader = self.make_dataloader(
            dataset=self.train_set,
            stage=sb.Stage.TRAIN,
            **self.train_loader_kwargs
        )
        self._loaded_checkpoint = False
        return dataloader
    
    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        if hasattr(self.hparams, 'gradient_accumulation'):
            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not finite
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.optimizer.zero_grad()

                # anneal lr every update
                # self.hparams.lr_annealing(self.optimizer)  #works with transformers and NoamScheduler
        else:
            loss.backward()
            if self.check_gradients(loss):
                self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        return super().evaluate_batch(batch, stage)

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # We always need to load a checkpoint on on_fit_start because we need to know
        # the `epoch_counter` value so that we know from which epoch to resume.
        # if True: #self.sorting not in getattr(self.train_set, 'CURRICULUM_KEYS', []):
        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                importance_key=self.ckpt_importance_key,
                device=torch.device(self.device)
            )
            self._loaded_checkpoint = True
        # NOTE: We use a different `improtance_key` function to make sure that 
        # the loaded checkpoint will be the one that has the full `final_sortings`
        # dictionary. This will probably take a lot of time since the function
        # checks all checkpoints and tries to find the one with the most
        # datapoints in `final_sortings`.
        # logger.info(f"On fit start. Type of train_set: {type(self.train_set)}.")
    
    @checkpoint_wrapper_cl
    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.on_valid_test_stage_start(stage)
        if stage != sb.Stage.TRAIN:
            return
        
        # ############################################################
        # ###################### CASE 1: No CL #######################
        # ############################################################
        if self.sorting in ["no", False, ""]:
            return  # no sorting needs be done
        
        # ############################################################
        # ################### CASE 2: Random CL ######################
        # ############################################################
        if self.sorting == "random":
            if not (
                isinstance(self.train_set, DataLoader)
                or isinstance(self.train_set, LoopedLoader)
            ):
                self.train_set = self.make_random_dataloader()
            # logger.info(f"On stage start: After random. Type of train_set: {type(self.train_set)}.")
            return self.train_set      
        
        # ############################################################
        # ############ CASE 3: Adaptive Pacing Function ##############
        # ############################################################
        if self.do_adaptive_pacing and len(self.sorting_dict) < len(self.current_trainset):
            logger.warning(f"Using an adaptive pacing function is not possible\
                with an empty or non-full sorting dictionary ({len(self.sorting_dict)=}...{len(self.current_trainset)=}).")
        if self.do_adaptive_pacing and len(self.sorting_dict) == len(self.current_trainset):
            logger.info(f"Using and adaptive pacing function on a sorting dictionary of length: {len(self.sorting_dict)}.")
            train_set = self.adaptive_pacing(
                n_epochs=self.hparams.number_of_epochs,
                epochs_per_group=self.epochs_per_adaptive_group,
                incremental=self.incremental_adaptive_pacing,
                normalize=False,
                inplace_norm=False
            )
            dataloader = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **self.train_loader_kwargs
            )
            return dataloader  
        
        # ############################################################
        # ####### CASE 3: Make sure there is a need to re-sort #######
        # ############################################################
        if (self.current_epoch % self.curriculum_update_every) > 0 and \
            (self.sorting in CurriculumDataset.CURRICULUM_KEYS) and \
            (len(self.current_trainset) > 0) and (len(self.sorting_dict) > 0):
            logger.info(f"Skipping trainset re-ordering.")
            train_set = self.current_trainset.filtered_sorted(
                sort_key=self.sorting,
                sorting_dict=self.sorting_dict,
                reverse=getattr(self.hparams, "reverse", False),
                noise_percentage=getattr(self.hparams, 'noisy_random_percentage', None),
            )
            dataloader = self.make_dataloader(train_set, stage=sb.Stage.TRAIN, **self.train_loader_kwargs)
            return dataloader
        
        # ############################################################
        # ################### CASE 4: Subsampling ####################
        # ############################################################
        # 'random' is not used with subsampling since it a special case where subsample_percentage=1
        if self.do_subsample and self.sorting != "random":
            logger.info("`do_subsample` is True.")
            percentage = getattr(self.hparams, 'subsampling_percentage', 0.3)
            increase_factor = getattr(self.hparams, 'subsampling_increase_factor', None)
            increase_type = getattr(self.hparams, 'subsampling_increase_type', 'additive')
            step_length = getattr(self.hparams, 'subsampling_step_length', 5)
            logger.info(f"On stage start: Before subsample. Type of train_set: {type(self.train_set)}.")
            sub_dataloader = self.subsample_trainset(
                percentage=percentage,
                increase_factor=increase_factor,
                increase_type=increase_type,
                step_length=step_length,
                ckpt_prefix="not-training-",
            )
            logger.info(f"On stage start: After subsample. Type of train_set: {type(self.train_set)}.")
            if sub_dataloader is not None:  # it is a tuple at this point.
                return sub_dataloader
            else:
                msg = f"`do_subsample` is true but not used."
                logger.warning(msg)
                raise Exception(msg)
        if self.do_subsample:
            # Add a dummy dataloader which will start the counter from zero when `curriculum_sort` is called.
            # This is because `curriculum_sort` tries to load the best current checkpoint 
            # which we already have from the first few (default) epochs. The "default" epoch
            # results don't have the not-training-TRAIN dataloader (which is used only for
            # creating and saving the subsampled datasets).
            if self.checkpointer is not None:
                ckpt_prefix="not-training-"
                self.make_dataloader(self.train_set, sb.Stage.TRAIN, ckpt_prefix, **self.train_loader_kwargs)
                # for _ in dummy_dataloader: break  # initialize iterators
                # self.checkpointer.add_recoverable(ckpt_prefix + stage.name, self.optimizer)
        
        
        # ############################################################
        # ######### CASE 5: Transfer CL with fixed sortings ##########
        # ############################################################
        if self.use_fixed_sorting:
            # The train dataloader must have already been created in train.py
            # logger.info("Ignoring on_stage_start since we assume that we already have a dataloader")
            return

        
        self._curriculum_log_saved = False
        def default_sort():
            reverse = getattr(self.hparams, 'reverse', False)
            train_set = self.train_set.filtered_sorted(
                sort_key="duration", 
                reverse=reverse,
                noise_percentage=getattr(self.hparams, 'noisy_random_percentage', None),
            )
            self.hparams.dataloader_options["shuffle"] = self.train_loader_kwargs["shuffle"] = False
            dataloader = self.make_dataloader(train_set, stage=sb.Stage.TRAIN, **self.train_loader_kwargs)
            # logger.info(f"On stage start: After default sort. Type of train_set: {type(self.train_set)}.")
            return dataloader

        def load_and_sort():
            assert isinstance(self.sorting_dict, dict) and len(self.sorting_dict) > 0, f"{type(self.sorting_dict)=}"
            self.sorting_dict.pop('num_datapoints', None)
            self.hparams.dataloader_options["shuffle"] = self.train_loader_kwargs["shuffle"] = False
            sb.dataio.dataset.set_output_keys([self.train_set], self.train_set.pipeline.output_mapping,)
            reverse = getattr(self.hparams, 'reverse', False)
            if len(self.final_sortings) == 0:
                raise Exception("You should have already loaded a checkpoint at this point.")
            if self.use_fixed_sorting:
                assert len(self.final_sortings) >= len(self.train_set), f"{len(self.final_sortings)=} --- {len(self.train_set)=}"
            elif len(self.sorting_dict) == len(self.train_set):
                # no need to copy since `self.sorting_dict = {}` will lose the connection (?? doubt??)
                self.final_sortings = self.sorting_dict.copy()
            # logger.debug(f"{len(self.final_sortings)=}, {len(self.train_set)=}.")
            if len(self.final_sortings) < len(self.train_set):
                raise Exception(f"This can't happen: {len(self.final_sortings)=} === {len(self.sorting_dict)=} === {len(self.train_set)=}.")
            dataset = self.train_set.filtered_sorted(
                sort_key=self.sorting, 
                sorting_dict=self.final_sortings,
                reverse=reverse,
                noise_percentage=getattr(self.hparams, 'noisy_random_percentage', None),
            )
            dataloader = self.make_dataloader(
                dataset=dataset,
                stage=sb.Stage.TRAIN,
                # ckpt_prefix="curriculum-",
                **self.train_loader_kwargs
            )
            # TODO: Maybe we don't need to empty these dicts
            self.train_set.sorting_dict = self.sorting_dict = {}
            self.final_sortings = {}
            # logger.info(f"On stage start: After load and sort. Type of train_set: {type(self.train_set)}.")
            return dataloader
        
        # ############################################################
        # ######## CASE 6: Metadata-based Scoring Functions ##########
        # ############################################################
        if self.sorting not in CurriculumDataset.CURRICULUM_KEYS:
            # it will only be saved on the first epoch
            self._save_curriculum_log(stage, epoch, self.sorting_dict)
            # The train data loader contains the already sorted data set
            return
        # ############################################################
        # ####### CASE 7: Default Ascending Sorting (for CL) #########
        # ############################################################
        # The parameter 'default_sorting_epochs' denotes the number of "preparation" epochs before
        # starting to sort by the curriculum key. A higher number will result in better prepared
        # models that can output better wer/cer/loss-based sortings. The drawback is that a high
        # number of preparation epochs will degrade the importance of our curriculum methods.
        if self.use_default_training:
            logger.info(f"Epoch {epoch}: Sorting the dataset based on the duration of the examples before the first epoch.")
            return default_sort()
        
        # ############################################################
        # ################# CASE 8: Loss Sorters #####################
        # ############################################################
        if self.sorting in getattr(self.train_set, 'LOSS_SORTERS', []):
            if len(self.sorting_dict) > 0:
                logger.debug(f"Curriculum Learning: Using the sorting dictionary based on the {self.sorting} method.")
                return load_and_sort()
            # If we get here then we do ascending duration-based sorting (default behaviour).
            # This shall only happen once (in the first epoch)
            logger.warning("Default sorting, even though the sorting method is: {}.".format(self.sorting))
            return default_sort()
        
        # ############################################################
        # ################# CASE 9: Metric Sorters ###################
        # ############################################################
        if self.sorting in getattr(self.train_set, 'METRIC_SORTERS', []):
            if len(self.sorting_dict) > 0:
                logger.debug(f"Curriculum Learning: Using the sorting dictionary based on the {self.sorting} method.")
                return load_and_sort()
            # If we get here then we do ascending duration-based sorting (default behaviour).
            # This shall only happen once (in the first epoch)
            logger.warning("Default sorting, even though the sorting method is: {}.".format(self.sorting))
            return default_sort()
        
        # ############################################################
        # ####### CASE 10: Joint Curriculum (NOT IMPLEMENTED) ########
        # ############################################################
        # For JointCurriculum (check src/classes.py)
        if isinstance(getattr(self.train_set, 'sorting', None), tuple) \
          and not getattr(self.train_set, 'sorting', [''])[0] in self.train_set.CURRICULUM_KEYS:
            logger.warn("Tuple sorting method is not yet implemented.")
            raise NotImplementedError("JointCurriculum has not been implemented.")
        raise Exception("UnexpectedException: You shouldn't be here.")

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.TRAIN:
            if self.sorting in CurriculumDataset.CURRICULUM_KEYS:
                self._loaded_checkpoint = False
                if self.current_epoch+1 <= getattr(self.hparams, 'default_sorting_epochs', 1):
                    # E.g. if we are on epoch 1 and `default_sorting_epochs` is 2, then 
                    # the sorting_dict will be empty.
                    # But, if we are on epoch 2 and `default_sorting_epochs` is 2, then
                    # the sorting dict must be filled with all scores, so that when we
                    # reach `on_stage_start` on epoch 3, we will have the sortings.
                    pass
                elif self.use_fixed_sorting is True:  # train_set is a dataloader
                    pass
                elif self.do_subsample and hasattr(self, 'train_subset'):
                    assert len(self.sorting_dict) == len(self.train_subset), f"{len(self.sorting_dict)=} ... {len(self.train_subset)=}"
                else:
                    assert len(self.sorting_dict) == len(self.train_set), f"{len(self.sorting_dict)=} ... {len(self.train_set)=}"
                if not (self.use_fixed_sorting and self.do_subsample):
                    self.final_sortings = self.sorting_dict.copy()
                elif len(self.final_sortings) < len(self.train_set) and self.do_subsample and self.use_fixed_sorting:
                    raise ValueError(f"You are doing subsampling with fixed sortings but the final sorting\
                        dictionary does not contain the full sortings ({len(self.final_sortings)=} while\
                            {len(self.train_set)=} and {len(self.sorting_dict)=}).")
            if self.sorting != "random":
                self._save_curriculum_log(stage, epoch, self.sorting_dict)
            # Also log the current training time
            logger.info(f"Currently training for {round(self.current_training_time/60, 2)} minutes.")
        # logger.info(f"On stage end. Type of train_set: {type(self.train_set)}.")
        return super().on_stage_end(stage, stage_loss, epoch)
    
    # special method for the seq + ctc loss curriculum sorting.
    def compute_loss(self, predictions, batch, stage=sb.Stage.TRAIN, reduction='mean', weight=None):
        # logger.info(f"Processing batch: {[i in self.sorting_dict for i in batch.id]}")
        # Get losses for each datapoint in the batch
        cw = weight or self.hparams.ctc_weight or 0.
        loss = self._compute_seq_loss(
            predictions, batch, stage=stage, 
            reduction=reduction
        )
        # Ctc-loss is calculated only in certain cases and only in the training stage
        if self.is_ctc_active(stage):
            loss_ctc = self._compute_ctc_loss(predictions, batch, stage, reduction=reduction)
            loss *= 1 - self.hparams.ctc_weight
            loss += cw * loss_ctc
        if self.sorting == "seq_ctc_loss":  # update dictionary
            # assert len(loss) == len(batch.id), "While loss was: {} (loss_seq={}, batch_id={}).".format(loss, loss, batch.id)
            loss = self._loss_curriculum_update(stage, batch, loss, predictions)
        return loss

    def adaptive_pacing(self,
        n_epochs: int,
        epochs_per_group: int,
        incremental: bool = True,
        normalize: Optional[bool] = True,
        inplace_norm: Optional[bool] = False
    ):
        """
        Arguments:
            n_epochs: For how many epochs shall these difficulties be used. If we are
                using a metadata-based curriculum learning method then this should be
                equal to the total number of epochs. Otherwise, it should equal the 
                number of subsampling epochs (check `subsampling_n_epochs` in the yaml files.)
            epochs_per_group: On how many epochs should each group be used for training?
                E.g. if 2, then the easiest group will be used for 2 epochs, then the
                     next group will be used for the next 2 epochs, and so on.
            incremental: If true then each subsequent subset will also contain the easy 
                examples taken from the previous subset. Check CurriculumDataset.split_into_k for more.
            normalize: Whether or not the sorting dictionary should be normalized. Notice that
                this normalization is IN-PLACE if inplace is True. The same normalization happens in
                CurriculumDataset._curriculum_filtered_ids
            inplace_norm: If true and `normalize` is also true then the sorting dictionary will
                be normalized in place (previous values will be overwritten).
        """
        if not isinstance(self.train_set, CurriculumBase):
            raise NotImplementedError(f"Cannot use adaptive pacing with metadata-based workflow. {type(self.train_set)=}")
        if len(self.sorting_dict) == 0:
            raise ValueError("The length of the sorting dictionary cannot be 0 when using an adaptive pacing function.")
        if normalize and not inplace_norm:
            sd = self.sorting_dict.copy()
        else:
            sd = self.sorting_dict
        # E.g. if 15 epochs and we train on each subset for 4 epochs, then we need
        #      15//4=3 groups. In the first 4 epochs we will have the 1st group,
        #      In epochs 5-8, we will have the 2nd group, in epochs 9-12 the 3rd group
        #      and in epochs 13-15 we will keep processing the 3rd group.
        n_difficulty_groups = min(1, n_epochs % epochs_per_group) + n_epochs // epochs_per_group
        try:
            train_set = self.train_set.adaptive_pacing(
                sd, 
                n_difficulty_groups,
                epochs_per_group,
                incremental,
                normalize,
                self.reverse,
                current_epoch=self.current_epoch,
            )
        except Exception as e:
            logger.info("Error occurred when applying the pacing function.")
            logger.info(f"The length of the current sorting dictionary is: {len(sd)}.")
            logger.info(f"Other hyperparameters: {n_difficulty_groups=} -- {epochs_per_group=} -- {incremental=} -- {normalize=} -- {self.reverse=} -- {self.current_epoch=}.")
            if len(sd) > 0:
                key1 = list(sd.keys())[0]
                logger.info(f"Sample from the sorting dictionary {key1}: {sd[key1]}.")
            raise e
        return train_set

    def subsample_trainset(
      self, 
      percentage: Optional[float] = None, 
      increase_factor: Optional[float] = None,
      increase_type: Optional[str] = "additive",
      step_length: Optional[int] = 10,
      ckpt_prefix="not-training-",
    ):
        assert self.hparams.do_subsample is True
        if not isinstance(percentage, float):
            percentage = getattr(self.hparams, 'subsampling_percentage', 0.3)
        # After how many epochs should we update the train set
        update_every = getattr(self.hparams, 'subsampling_n_epochs', 5)
        dummy_epoch = self.current_epoch//update_every + 1
        calculated_dict = False
        if isinstance(increase_factor, float) and increase_factor > 0:
            if increase_type in ['additive', '+']:
                percentage = round(min(1.0, percentage+(increase_factor)*(dummy_epoch-1)), 4)
            elif increase_type in ['multiplicative', '*', "exp", "exponential"]:
                # E.g. if initial percentage is 0.1, increase_factor=1.5 and step_length=5
                #      on 1st epoch: percentage *= 1.5^(0/5) = 0.1*1 = 0.1
                #      on 5th epoch: percentage *= 1.5^(4/5) = 0.1*1.383 = 0.12
                #      on 30th epoch: percentage *= 1.5^(29/5) = 0.1*10.5 = 1.05 -> 1.0
                assert increase_factor > 1, "Multiplicative should be above 1 since eitherwise the trainset will get shrinked to 0"
                epoch_to_use = max(update_every * (dummy_epoch-1), 2) - 1  # e.g. 1, 4, 9, 14...
                percentage = round(min(1, percentage*increase_factor**(epoch_to_use/step_length)), 4)
                logger.info(f"Subsampling using {percentage*100}% of the training set.")
            suffix = f"percentage={percentage}"
        else:
            suffix = f"epoch={dummy_epoch}"
        log_dir = os.path.join(self.hparams.output_folder, 'curriculum_logs')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        curr_log_path = os.path.join(
            self.hparams.output_folder, 
            'curriculum_logs', 
            f"{self.dict_log_prefix}{self.sorting}_dict-{suffix}.log"
        )
        subset_ids_path = os.path.join(
            self.hparams.output_folder,
            'curriculum_logs',
            f"{self.dict_log_prefix}{self.sorting}_subset_ids-{suffix}.npy"
        )
        def get_train_subset(shuffled_train_ids):
            if self.use_default_training:
                train_set = self.train_set.filtered_sorted(
                    sort_key="duration",
                    noise_percentage=getattr(self.hparams, 'noisy_random_percentage', None),
                )
            else:
                train_set = self.train_set
            if isinstance(train_set, FilteredSortedDynamicItemDataset) or isinstance(train_set, FilteredSortedFrequencyCL):
                DatasetClass, kwargs = FilteredSortedDynamicItemDataset, {}
                if isinstance(train_set, FilteredSortedFrequencyCL):
                    DatasetClass, kwargs = FilteredSortedFrequencyCL, train_set.kwargs
                logger.info(f"Using curriculum subset for filteredsorted dataset. {self.use_default_training=}")
                # train_subset = CurriculumSubset(self.train_set, shuffled_train_ids)
                shuffled_train_ids = [train_set.data_ids[i] for i in shuffled_train_ids]
                train_subset = DatasetClass(train_set, shuffled_train_ids, **kwargs)
            elif isinstance(train_set, DataLoader):
                # For random sorting
                raise NotImplementedError(f"Cannot subsample a dataloader train set of type {type(self.train_set)} ({self.sorting=}).")
            elif isinstance(train_set, CurriculumDataset):
                train_subset = CurriculumSubset(train_set, shuffled_train_ids)
            else:
                raise TypeError(f"Cannot subsample train set of type {type(train_set)} ({self.sorting=}).")
            return train_subset
        # logger.info(f"{curr_log_path=}")
        if percentage >= 1 and self.use_fixed_sorting:
            if len(self.final_sortings) > len(self.sorting_dict):
                self.sorting_dict = self.final_sortings.copy()
            else:
                # sorting_dict and final_sortings are filled.
                # The epoch0 dictionary must exist since it is created 
                # before the 1st epoch.
                self.load_sorting_dict(epoch=0, inplace=True)
            shuffled_train_ids = np.arange(0, len(self.train_set))
            self.train_subset = get_train_subset(shuffled_train_ids)
        elif os.path.isfile(curr_log_path) and os.path.isfile(subset_ids_path):
            logger.info("Subsampling: Loading pre-existing CL dictionary.")
            self.sorting_dict = self.load_sorting_dict(sorting_dict_log=curr_log_path, inplace=False)
            if len(self.final_sortings) < len(self.sorting_dict) and self.use_fixed_sorting:
                self.final_sortings = self.load_sorting_dict(epoch=0, inplace=False)
            shuffled_train_ids = np.load(subset_ids_path)
            self.train_subset = get_train_subset(shuffled_train_ids)
            logger.info(strip_spaces(f"Loaded a precomputed train set with \
                {len(shuffled_train_ids)} datapoints (out of {len(self.train_set)})."))
        else:
            shuffled_train_ids = np.random.permutation(len(self.train_set))[:round(percentage*len(self.train_set))]
            self.train_subset = get_train_subset(shuffled_train_ids)
            assert hasattr(self.train_subset, 'filtered_sorted'), f"{type(self.train_subset)=}"
            if self.current_epoch == 1 and len(self.sorting_dict) > 0:
                logger.info(f"Selecting {len(shuffled_train_ids)} random ids.")
                self.sorting_dict = {k: self.sorting_dict[k] for k in self.train_subset.data_ids}
            elif self.use_fixed_sorting and len(self.final_sortings) > len(self.sorting_dict):
                logger.info(f"Selecting {len(shuffled_train_ids)} random ids from the fixed sortings.")
                self.sorting_dict = {k: self.final_sortings[k] for k in self.train_subset.data_ids}
            else:
                logger.info("We are going to create a new sorting dictionary.")
                if (self.sorting in CurriculumDataset.CURRICULUM_KEYS) and not self.use_default_training:
                    self.sorting_dict = self.create_curriculum_dict(
                        self.train_subset,
                        sorting_dict_save_path=curr_log_path,
                        update_final_dict=not self.use_fixed_sorting,
                        ckpt_prefix=ckpt_prefix,
                        try_recover=False,  # always re-iterate the dataloader so that the sorting-dict will be calculated from scratch
                    )
                    calculated_dict = True
                    logger.info(f"Created a curriculum dictionary of size: {len(self.sorting_dict)=}.")
                    if len(self.sorting_dict) > len(shuffled_train_ids):
                        logger.warning(f"CREATED A CURRICULUM OF SIZE {len(self.sorting_dict)} WHILE THE SUBSET SHOULD BE OF LENGTH {len(shuffled_train_ids)}.")
                        self.sorting_dict = {k: v for k, v in self.sorting_dict.items() if k in self.train_subset.data_ids}
            np.save(subset_ids_path, shuffled_train_ids)
            logger.info(strip_spaces(f"Calculated a new train set with \
                {len(self.train_subset)} datapoints (out of {len(self.train_set)})."))
        if (self.sorting in BaseASR.DURATION_CL_KEYS) or self.use_default_training:
            logger.info("Duration-based sorting + subsampling. This implies that the `noisy` CL method cannot be used.")
            train_set = self.train_subset.filtered_sorted(
                sort_key="duration",
                reverse=True if self.sorting == "descending" else False,
            )
        elif self.use_transfer_cl and isinstance(self.train_subset, FilteredSortedDynamicItemDataset):
            train_set = self.train_subset
        else:
            train_set = self.train_subset.filtered_sorted(
                sort_key=self.sorting,
                sorting_dict=self.sorting_dict,
                reverse=self.reverse,
                noise_percentage=getattr(self.hparams, 'noisy_random_percentage', None),
            )
        logger.info(f"Size of the 'subsampled' trainset: {len(train_set)} and type: {type(train_set)} \
            \nwhile type of train_subset: {type(self.train_subset)=}.")
        dataloader = self.make_dataloader(train_set, stage=sb.Stage.TRAIN, **self.train_loader_kwargs)
        if not self.use_fixed_sorting:
            self.final_sortings = {}
        logger.info(f"Sub dataloader of length: {len(dataloader)}")
        return (dataloader, calculated_dict)


    def is_ctc_active(self, stage):
        """Check if CTC is currently active.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        """
        if stage != sb.Stage.TRAIN:
            return False
        return self.current_epoch <= self.hparams.number_of_ctc_epochs

    def prepare_features(self, stage, wavs, ids=None):
        """Prepare features for computation on-the-fly

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        wavs : tuple
            The input signals (tensor) and their lengths (tensor).
        """
        if isinstance(wavs, tuple) and len(wavs) == 2:
            wavs, wav_lens = wavs
        else:
            raise NotImplementedError(f"{len(wavs)=}, \n{wavs=}")

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

            if hasattr(self.hparams, "augmentation"):  # NOT SpecAugment
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Feature computation and normalization
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        return feats, wav_lens

    def prepare_tokens(self, stage, tokens):
        """Double the tokens batch if features are doubled.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        tokens : tuple
            The tokens (tensor) and their lengths (tensor).
        """
        tokens, token_lens = tokens
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens], dim=0)
        return tokens, token_lens
    
    def load_sorting_dict(
            self, 
            sorting_dict_log: str = None, 
            from_path: str = None, 
            epoch: int = None,
            inplace: bool = True,
        ) -> dict:
        """ Loads a sorting dict for curriculum learning.
            If from_path is None we will use <output_folder>/curriculum_logs/<cl-method>_dict-epoch=<epoch>.log
            If epoch is also None then we will use <max-epoch> instead.
        """
        if sorting_dict_log is None:
            if from_path is None:
                from_path = os.path.join(self.hparams.output_folder, "curriculum_logs")
            assert os.path.isdir(from_path), f"Could not locate: {from_path}"
            if epoch is None:
                try:
                    # Get the log that corresponds to the biggest epoch
                    epoch = max(map(int, map(lambda filename: filename.split("epoch=")[1].split(".")[0], os.listdir(from_path))))
                except IndexError:
                    msg = f"Expected log file of the form /path/to/<prefix><cl-method>_dict-epoch=<epoch>.log \
                        but we couldn't split on the '=' and '.' symbols. The files we checked were: \
                        {os.listdit(from_path)}."
                    raise IndexError(msg)
                except ValueError:
                    msg = f"Could not find a valid integer as the epoch number under: {os.listdir(from_path)}."
                    raise ValueError(msg)
            sorting_dict_log = os.path.join(from_path, f"{self.dict_log_prefix}{self.sorting}_dict-epoch={epoch}.log")
        assert os.path.isfile(sorting_dict_log), f"Could not locate the presumed sorting_dict log file: {sorting_dict_log}"
        sd = load_sorting_dictionary(sorting_dict_log)
        if inplace:
            self.sorting_dict = sd
            self.final_sortings = sd.copy()
            return self.sorting_dict
        else:
            return sd
    
    def create_curriculum_dict(
        self, 
        train_set: Optional[DynamicItemDataset] = None, 
        sorting_dict_save_path: Optional[str] = None, 
        progressbar: Optional[bool] = None,
        update_final_dict: bool = True,
        ckpt_prefix: str = "dataloader-",
        try_recover: bool = True,
    ):
        """ Create and return a curriculum dictionary based on the current checkpoint.
            Loads a checkpoint and iterates the train set for one epoch in order to 
            create the sorting dictionary.
            Returns the curriculum without updating any parameters.
        """
        if train_set is None:
            train_set = self.train_set
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()
        # Wrap modules with parallel backend after jit
        self._wrap_distributed()
        # Eval mode since we don't want to update the loss
        self.modules.eval()
        # Check progressbar config
        if progressbar is None:
            progressbar = not self.noprogressbar
        enable = progressbar and sb.utils.distributed.if_main_process()
        # Create dataloader for the training set (we don't care about order)
        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, 
                stage=sb.Stage.TRAIN, 
                ckpt_prefix=ckpt_prefix, #"not-training-",
                **self.train_loader_kwargs
            )
        # Load a checkpoint if we previously stopped midway
        if try_recover and self.checkpointer is not None:
            old_epoch = self.current_epoch
            self.checkpointer.recover_if_possible(
                importance_key=self.ckpt_importance_key,
                device=torch.device(self.device)
            )
            self.current_epoch = old_epoch
        # self.checkpointer.recover_if_possible(device=torch.device(self.device))
        self.sorting_dict = {}
        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        if self.use_fixed_sorting:
            logger.info(f"Will process {len(train_set.dataset)} data points from transfer learning CL.")
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
        ) as t:
            for batch in t:
                self.step += 1
                out = self.compute_forward(batch, stage=sb.Stage.TRAIN)
                loss = self.compute_loss(
                    out,
                    batch,
                    stage=sb.Stage.TRAIN,
                    reduction="mean",
                    weight=self.hparams.ctc_weight
                )

                # Debug mode only runs a few batches
                if self.debug:# and self.step == self.debug_batches:
                    break

                if (
                    self.checkpointer is not None
                    and self.ckpt_interval_minutes > 0
                    and time.time() - last_ckpt_time
                    >= self.ckpt_interval_minutes * 60.0
                ):
                    # This should not use run_on_main, because that
                    # includes a DDP barrier. That eventually leads to a
                    # crash when the processes'
                    # time.time() - last_ckpt_time differ and some
                    # processes enter this block while others don't,
                    # missing the barrier.
                    if sb.utils.distributed.if_main_process():
                        self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()
        # assert len(self.sorting_dict) == len(self.train_set), f"{len(self.sorting_dict)=}, {len(self.train_set)=}"
        dataset = self.train_subset if self.do_subsample and hasattr(self, 'train_subset') else self.train_set
        assert len(self.sorting_dict) == len(dataset), f"{len(self.sorting_dict)=}, {len(dataset)=}"
        if sorting_dict_save_path is not None:
            save_sorting_dictionary(self.sorting_dict, sorting_dict_save_path)
            logger.info("Log file saved under: {}".format(sorting_dict_save_path))
            self._curriculum_log_saved = True
        if update_final_dict:
            self.final_sortings = self.sorting_dict.copy()
        return self.sorting_dict

    def _detokenize_from_list(self, predicted_tokens):
        if getattr(self.hparams, 'use_lm', False) is True:
            assert hasattr(self.tokenizer, 'decode_ids'), type(self.tokenizer)
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
        else:
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )
        return predicted_words
    
    def _valid_test_objectives(self, batch, predicted_tokens, stage):
        assert stage != sb.Stage.TRAIN, stage
        predicted_words = self._detokenize_from_list(predicted_tokens)
        # target_words = [words.split(" ") for words in batch.trn]
        tokens, tokens_lens = batch.tokens
        target_words = undo_padding(tokens, tokens_lens)
        target_words = self._detokenize_from_list(target_words)
        # if random.random() > 0.99:
        #     print("  preds-truth pairs:", list(zip(predicted_words, target_words))[-1])

        # Monitor word error rate and character error rated at
        # valid and test time.
        # self.wer_metric.append(batch.__key__, predicted_words, target_words)
        # self.cer_metric.append(batch.__key__, predicted_words, target_words)
        self.wer_metric.append(batch.id, predicted_words, target_words)
        self.cer_metric.append(batch.id, predicted_words, target_words)
    

    def _save_curriculum_log(self, stage, epoch: int, sorting_dict: dict):
        if self.sorting not in self.VALID_CL_KEYS or len(self.sorting) == 0:
            return
        if self._curriculum_log_saved is True or stage != sb.Stage.TRAIN or (not self.sorting_dict_needs_save):
            return
        def __save__(sorting_dict_log, ordered_examples):
            with open(sorting_dict_log, 'w') as fa:
                # fa.write("ID\tScore\n")
                fa.write('\n'.join(ordered_examples))
            logger.info("Log file saved under: {}".format(sorting_dict_log))
            self._curriculum_log_saved = True
        if self.sorting in CurriculumDataset.CURRICULUM_KEYS:
            save_folder = os.path.join(self.hparams.output_folder, "curriculum_logs")
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
            # assert hasattr(self, 'sorting_dict')
            sorting_dict_log = os.path.join(save_folder, f"{self.dict_log_prefix}{self.sorting}_dict-epoch={epoch}.log")
        elif isinstance(self.train_set, (FilteredSortedDynamicItemDataset, FilteredSortedFrequencyCL)) \
          and self.sorting != "random":
            # Output path
            sorting_dict_log = os.path.join(self.hparams.output_folder, f"{self.dict_log_prefix}{self.sorting}_dict.log")
            if os.path.isfile(sorting_dict_log):
                return
        ordered_examples = self._get_ordered_examples(sorting_dict)
        __save__(sorting_dict_log, ordered_examples)
    
    def _get_ordered_examples(self, sorting_dict):
        if self.sorting in CurriculumDataset.CURRICULUM_KEYS:
            # assert hasattr(self, 'sorting_dict')
            ordered_examples = [f"{k}\t{v}" for k, v in sorted(sorting_dict.items(), key=lambda x: x[1], reverse=True)]
        elif isinstance(self.train_set, (FilteredSortedDynamicItemDataset, FilteredSortedFrequencyCL)) \
          and self.sorting != "random":
            if self.sorting in FrequencyCL.VALID_FREQUENCY_TYPES:
                ordered_dict = self.train_set.get_frequencies()
                ordered_examples = [f"{key}\t{value}" for key, value in sorted(ordered_dict.items(), key=lambda x: x[1], reverse=True)]
            else:
                # These are the keys in sorted order
                data_ids = self.train_set.data_ids
                sort_key = "duration"
                temp_keys = (set([] if self.sorting is None else [sort_key]))
                filtered_ids = []
                with self.train_set.output_keys_as(temp_keys):
                    for i, data_id in enumerate(data_ids):
                        data_point = self.train_set.data[data_id]
                        data_point["id"] = data_id
                        computed = self.train_set.pipeline.compute_outputs(data_point)
                        # Add (main sorting index, current index, data_id)
                        # So that we maintain current sorting and don't compare
                        # data_id values ever.
                        filtered_ids.append((computed[sort_key], i, data_id))
                # Here we will only save the ids of the sorted dataset since
                # we don't have any values.
                # `reverse` is always true so it doesn't matter if we have ascending
                # or descneding-based curriculum.
                ordered_examples = [f"{tup[2]}\t{tup[0]}" for tup in sorted(filtered_ids, reverse=True)]
        else:
            raise NotImplementedError(f"Could not save sorting dict for method {self.sorting}\
                when the type of the train set is {type(self.train_set)}.")
        return ordered_examples

    def _update_dict(self, batch_id, value, dur, sorting_method=None):
        if not isinstance(value, tuple):
            value = (value, dur)
        self.sorting_dict[batch_id] = value
        if self.debug:
            logger.debug(
                f"id: {batch_id} - {sorting_method or self.sorting}: "
                f"- duration: {dur} - value: "
                f"{self.sorting_dict[batch_id]}"
            )

    def _metric_curriculum_update(self, stage, batch, loss, predictions, sorting_method=None):
        sorting_method = sorting_method or self.sorting
        if stage != sb.Stage.TRAIN or self.current_epoch+1 <= getattr(self.hparams, 'default_sorting_epochs', 1):
            return loss
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("Expected 'tokenizer' to exist for Metric-based Curriculum Learning."\
                " Check the asr_model.py file in the default recipe to see how this"\
                    "should be implemented.")
        assert sorting_method in CurriculumDataset.METRIC_SORTERS, "This must be true!"
        keep_confs = getattr(self.hparams, 'keep_confs', False)
        if keep_confs:
            get_value = lambda wer, conf, dur: (wer, conf, dur)
        else:
            get_value = lambda wer, conf, dur: wer
        # These should be the indices of p_tokens, scores in the asr_model.py file
        # in the compute_forward function.
        if sorting_method == 'cer':
            # convert the sequence of words to a sequence of characters.
            # this way we will get CER instead of WER
            postprocess = lambda word_seq: [list('_'.join(w)) for w in word_seq]
        elif sorting_method == 'wer':
            # Do nothing
            postprocess = lambda word_seq: word_seq
        predicted_tokens = [h[0] for h in predictions["tokens"]]
        assert len(predicted_tokens) == len(batch.id)
        tokens, tokens_lens = batch.tokens
        # Decode token terms to words
        predicted_words: list = self._detokenize_from_list(predicted_tokens)
        # Convert indices to words
        target_words = undo_padding(tokens, tokens_lens)
        target_words = postprocess(self._detokenize_from_list(target_words))

        # Process predictions and truth so that they don't contain special tokens (.br, .fr etc)
        predicted_words = [re.sub("\.\w+", "", ' '.join(txt)).split() for txt in predicted_words]
        target_words = [re.sub("\.\w+", "", ' '.join(txt)).split() for txt in target_words]
        details: List[dict] = wer_details_for_batch(batch.id, target_words, predicted_words, True)
        for dp_id in range(len(batch.id)):
            dur = batch.duration[dp_id].cpu().item()
            value = get_value(
                wer=round(details[dp_id]['WER'],4), 
                conf=round(predictions["scores"][dp_id].cpu().item(), 4),
                dur=dur
            )
            self._update_dict(batch.id[dp_id], value, dur)
        return loss

    def _loss_curriculum_update(self, stage, batch, losses, predictions, sorting_method=None):
        if stage != sb.Stage.TRAIN or self.current_epoch+1 <= getattr(self.hparams, 'default_sorting_epochs', 1):
            return self._mean_redact(losses, batch)
        sorting_method = sorting_method or self.sorting
        assert len(losses) == len(batch.id)
        assert sorting_method in CurriculumDataset.LOSS_SORTERS, "This must be true!"
        if getattr(self.hparams, "dur_normalize", False) is True:
            get_value = lambda dp_id, dur: (round(losses[dp_id].item(), 4), dur)
        else:
            get_value = lambda dp_id, dur: round(losses[dp_id].item(), 4)
        for dp_id in range(len(batch.id)):
            # Either in the format (loss_value, duration) or simply (loss_value)
            dur = batch.duration[dp_id].cpu().item()
            self._update_dict(batch.id[dp_id], get_value(dp_id, dur), dur)
        return self._mean_redact(losses, batch)
    
    def _compute_seq_loss(self, predictions, batch, stage=sb.Stage.TRAIN, reduction="mean", sorting_method=None):
        # NOTE: You will probably have to override this method.
        # logger.debug("Using the default implementation for seq_loss computation (`base_asr_model.py`).")

        # sorting_method = sorting_method or self.sorting
        if (sorting_method is None and \
            len(self.sorting_dict) < len(self.current_trainset)) or \
                not self.do_adaptive_pacing or \
                    stage != sb.Stage.TRAIN:
            sorting_method = self.sorting
        if not self.do_adaptive_pacing:
            assert sorting_method is not None, f"{sorting_method=}...{self.sorting=}...{len(self.sorting_dict)=}...{len(self.current_trainset)=}...{stage=}"
        # Computing seq2seq loss
        tokens_eos, tokens_eos_lens = self.prepare_tokens(
            stage, batch.tokens_eos
        )
        p_seq = predictions["seq_logprobs"]

        reduction = "mean" if reduction == "batch" else reduction
        postproc = lambda stage, batch, loss, predictions, sorting_dict: loss
        if stage != sb.Stage.TRAIN:
            pass
        elif self.use_fixed_sorting is True:
            pass
        elif (self.current_epoch % self.curriculum_update_every != 0) and (not self.use_default_training):
            # NOTE (IMPORTANT): If do_adaptive is true then when we pass the 
            # first self.curriculum_update_every epochs, we will end up updating
            # only certain entries of the sorting dictionary. I.e. At first the 
            # dictionary will have 100% of the data, then after the 1st epoch
            # the train set will be only e.g. 10% of the whole train set (due 
            # to the adaptive pacing), ..., at the curriculum_update_every-th
            # epochs the train set will still be a subset of the whole train set.
            # This means that after curriculum_update_every the sorting dict will
            # only update certain entries of it (the ones that are in the current
            # train set). And so a micxture of the old and new sortings will be mixed.
            pass
        elif sorting_method == "seq_loss":
            reduction = "batch"
            postproc = self._loss_curriculum_update
        elif sorting_method == "seq_ctc_loss":
            reduction = "batch"    
        elif sorting_method in CurriculumDataset.METRIC_SORTERS:
            postproc = self._metric_curriculum_update        
        
        #logging.info(f"Shape of p_seq (log probs): {p_seq.shape} === While shape of targets: {tokens_eos.shape}.")
        loss_seq = self.hparams.seq_cost(
            p_seq, 
            tokens_eos, 
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
            reduction=reduction
        )
        loss_seq = postproc(stage, batch, loss_seq, predictions, sorting_method)
        return loss_seq
    
    def _compute_ctc_loss(self, predictions, batch, stage=None, reduction="mean", sorting_method=None):
        # NOTE: You will probably have to override this method.
        # logger.debug("Using the default implementation for ctc_loss computation (`base_asr_model.py`).")
        # Computing ctc loss
        assert hasattr(self, 'feat_lens')
        if (sorting_method is None and \
            len(self.sorting_dict) < len(self.current_trainset)) or \
                not self.do_adaptive_pacing or \
                    stage != sb.Stage.TRAIN:
            sorting_method = self.sorting
        if not self.do_adaptive_pacing:
            assert sorting_method is not None, f"{sorting_method=}...{self.sorting=}...{len(self.sorting_dict)=}...{len(self.current_trainset)=}...{stage=}"
        # sorting_method = sorting_method or self.sorting
        # tokens, tokens_lens = batch.tokens
        tokens, tokens_lens = self.prepare_tokens(stage, batch.tokens)
        if len(predictions) == 1:
            # loss_ctc is None since we reached the max number of ctc epochs
            return 0.
        # NOTE: We assume that the metric-based CL will be handled on the _compute_seq_loss
        #   function which is always called.
        reduction = "mean" if reduction == "batch" else reduction
        postproc = lambda stage, batch, loss, predictions: loss
        if stage != sb.Stage.TRAIN:
            pass
        elif self.use_fixed_sorting is True:
            pass
        elif sorting_method == "ctc_loss":
            reduction="batch"
            postproc = self._loss_curriculum_update
        elif sorting_method == "seq_ctc_loss":
            reduction = "batch"
        
        p_ctc = predictions["ctc_logprobs"]
        # logger.info(f"{p_ctc.shape=}\n======\n{wav_lens=}")
        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, self.feat_lens, tokens_lens, reduction=reduction
        )
        loss_ctc = postproc(stage, batch, loss_ctc, predictions)
        return loss_ctc
    
    def _mean_redact(self, losses: torch.Tensor, batch):
        # Tries to immitate pytorch's 'mean' reduction method which we will 
        # not be able to use when we use Curriculum Learning with Loss Sorting
        # happening on on_stage_end (since during on_stage_end we will assume
        # that we have all the information we need for the loss values per 
        # example which means that these values must be extracted for each 
        # example while training (aka we cannot perform mean reduction 
        # because we will lose any kind of example-specific information))
        # return batch_values.mean()
        token_lens = batch.tokens[1]
        losses /= token_lens
        return losses.mean()

    # def _save_intra_sorting_ckpt(self):  # only used for curriculum learning
    #     if (self.sorting not in self.train_set.CURRICULUM_KEYS) or (self.use_fixed_sorting is True): 
    #         return
    #     """Saves a CKPT with specific intra-sorting flag."""
    #     self.checkpointer.save_and_keep_only(
    #         end_of_epoch=False,
    #         num_to_keep=2,
    #         # ckpt_predicate=lambda c: INTRA_SORTING_CKPT_FLAG in c.meta,
    #         # meta={INTRA_SORTING_CKPT_FLAG: True},
    #         verbosity=logging.DEBUG,
    #     )
    
    # def _on_curriculum_end(self, running_average, time_to_sort, epoch):  # only used for curriculum learning
    #     if self.sorting not in self.train_set.CURRICULUM_KEYS: return
    #     """Handles the end of curriculum sorting."""
    #     self.hparams.train_logger.log_stats(
    #         stats_meta={
    #             "epoch": epoch, 
    #             "average_curriculum_val": running_average,
    #             "time_to_sort": time_to_sort,
    #         },
    #     )
    #     self.checkpointer.save_and_keep_only(end_of_epoch=False)

    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        try:
            self.current_training_time = self.current_training_time +  time.perf_counter() - self.start_epoch_time
            save_dict = {
                "step": self.step,
                "avg_train_loss": self.avg_train_loss,
                "current_training_time": self.current_training_time,
            }
        except AttributeError:
            # i.e. self.current_training_time not found
            save_dict = {"step": self.step, "avg_train_loss": self.avg_train_loss,}
        if not self.sorting_dict_needs_save:
            pass
        elif self.current_epoch <= self.hparams.number_of_curriculum_epochs and \
              (isinstance(self.sorting_dict, dict) and len(self.sorting_dict) > 0):
            #   (self.sorting in getattr(self.train_set, 'CURRICULUM_KEYS', []))):
            # ###############################################
            # ########## For Curriculum Learning ############
            # ###############################################
            save_dict["curriculum_step"] = getattr(self.train_set, "step", 0)
            save_dict["running_average"] = getattr(self.train_set, "running_average", 0)
            save_dict['curriculum_dict'] = self.sorting_dict
            dataset = self.train_subset if self.do_subsample and hasattr(self, 'train_subset') else self.train_set
            if len(self.sorting_dict) == len(dataset) and not self.use_fixed_sorting:
                self.final_sortings = self.sorting_dict.copy()
            save_dict['final_sortings'] = self.final_sortings
            
        with open(path, "w") as w:
            w.write(yaml.dump(save_dict))
        self.start_epoch_time = time.perf_counter()

    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch, device):
        del end_of_epoch
        del device
        with open(path) as f:
            save_dict = yaml.load(f, Loader=YamlTupleSafeLoader)
        self.step = save_dict["step"]
        self.avg_train_loss = save_dict["avg_train_loss"]

        # Recover timing information
        self.start_epoch_time = time.perf_counter()
        self.current_training_time = save_dict.get("current_training_time", 0.0)

        if (
            self.sorting not in CurriculumDataset.CURRICULUM_KEYS
        ):
            return
        if (self.sorting in CurriculumDataset.CURRICULUM_KEYS and self.use_fixed_sorting is True):
            if len(self.sorting_dict) == 0:
                self.load_sorting_dict(epoch=0, inplace=True)
            return
        # ###############################################
        # ########## For Curriculum Learning ############
        # ###############################################
        self.train_set.step = save_dict.get("curriculum_step", 0)
        self.train_set.running_average = save_dict.get('running_average', 0.0)
        d = save_dict.get('curriculum_dict', {})
        if len(d) == 0:
            self.sorting_dict = {}
            return
        # default_sorting_epochs = getattr(self.hparams, 'default_sorting_epochs', 1)
        # if (self.sorting not in getattr(self.train_set, 'CURRICULUM_KEYS', [])) or\
        #   (self.hparams.epoch_counter.current <= default_sorting_epochs):
        #     return

        # # Uncomment when re-training older models
        # try:
        #     self.train_set.step = save_dict["curriculum_step"]
        # except KeyError:
        #     return
        self.sorting_dict = self.train_set.sorting_dict = d
        self.final_sortings = save_dict['final_sortings']
    
    def _save_intra_epoch_ckpt(self):
        """Saves a CKPT with specific intra-epoch flag."""
        meta = {sb.core.INTRA_EPOCH_CKPT_FLAG: True}
        if self.sorting in CurriculumDataset.CURRICULUM_KEYS:
            meta['n_sorted_datapoints'] = len(self.final_sortings)
        self.checkpointer.save_and_keep_only(
            end_of_epoch=False,
            num_to_keep=2,
            ckpt_predicate=lambda c: sb.core.INTRA_EPOCH_CKPT_FLAG in c.meta,
            meta=meta,
            verbosity=logging.DEBUG,
        )
    
    @staticmethod
    def ckpt_importance_key(ckpt):
        if 'n_sorted_datapoints' in ckpt.meta:
            # Faster. `n_sorted_datapoints` can be accessed since we have changed
            # the definition of `_save_insta_epoch_ckpt`.
            return (ckpt.meta['n_sorted_datapoints'], ckpt.meta["unixtime"])
        path_to_CKPT = ckpt.paramfiles['brain']
        with open(path_to_CKPT) as f:
            save_dict = yaml.load(f, Loader=YamlTupleSafeLoader)
        n_sorted_datapoints = len(save_dict.get('final_sortings', []))
        return (n_sorted_datapoints, ckpt.meta["unixtime"])
        
    def _profile(self, func, out_file='curriculum_sort.prof', *args, **kwargs):
        if not getattr(self, 'profiler', False):
            return  # no profiling will be performed
        import cProfile
        import pstats
        with cProfile.Profile() as pr:
            output = func(*args, **kwargs)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(filename=out_file)
        return output
