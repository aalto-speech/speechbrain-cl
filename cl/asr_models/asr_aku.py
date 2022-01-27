#!/usr/bin/env python3
import torch
import logging
import speechbrain as sb
from cl.curriculum import CurriculumDataset
from cl.base_asr_model import BaseASR


logger = logging.getLogger(__name__)

# Brain class for speech recognition training
class ASR_Aku(BaseASR):
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, 
                 checkpointer=None, sorting=None, train_set=None, tokenizer=None,
                 train_loader_kwargs=None, *args, **kwargs):
        super(ASR_Aku, self).__init__(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams, 
            run_opts=run_opts,
            checkpointer=checkpointer,
            sorting=sorting,
            train_set=train_set,
            train_loader_kwargs=train_loader_kwargs,
            *args, **kwargs,
        )
        self.tokenizer = tokenizer
    
    def compute_forward(self, batch, stage):
        """Runs all the computation of the CTC + seq2seq ASR. It returns the
        posterior probabilities of the CTC and seq2seq networks.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            At training time it returns predicted seq2seq log probabilities.
            If needed it also returns the ctc output log probabilities.
            At validation/test time, it returns the predicted tokens as well.
        """
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        feats, self.feat_lens = self.prepare_features(stage, batch.sig, batch.id)
        self.feat_lens = self.feat_lens.to(self.device)
        tokens_bos, _ = self.prepare_tokens(stage, batch.tokens_bos)

        # Running the encoder (prevent propagation to feature extraction)
        encoded_signal = self.modules.encoder(feats.detach())

        # Embed tokens and pass tokens & encoded signal to decoder
        embedded_tokens = self.modules.embedding(tokens_bos)
        # print(f"{encoded_signal.shape=} ====== {embedded_tokens.shape=}")
        if encoded_signal.shape[0] != embedded_tokens.shape[0]:
            encoded_signal = encoded_signal.reshape(embedded_tokens.shape[0], -1, encoded_signal.shape[-1]).to(self.device)
            assert encoded_signal.shape[0] == 1, "Batch size must be 1 in this case bcs otherwise we don't know how to reshape feat_lens"
            self.feat_lens = torch.Tensor([1.0]).to(self.device)
        # print(f"{embedded_tokens.device=}\n{encoded_signal.device=}\n{self.feat_lens.device=}")
        decoder_outputs, _ = self.modules.decoder(
            embedded_tokens, encoded_signal, self.feat_lens
        )

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(decoder_outputs)
        predictions = {"seq_logprobs": self.hparams.log_softmax(logits)}

        if stage == sb.Stage.TRAIN:
            if self.is_ctc_active(stage):
                # Output layer for ctc log-probabilities
                ctc_logits = self.modules.ctc_lin(encoded_signal)
                predictions["ctc_logprobs"] = self.hparams.log_softmax(ctc_logits)
            if self.sorting in CurriculumDataset.METRIC_SORTERS:
                predictions["tokens"], predictions['scores'] = self.hparams.valid_search(
                    encoded_signal, self.feat_lens
                )
        elif stage == sb.Stage.VALID:
            predictions["tokens"], predictions['scores'] = self.hparams.valid_search(
                encoded_signal, self.feat_lens
            )
        elif stage == sb.Stage.TEST:
            predictions["tokens"], predictions['scores'] = self.hparams.test_search(
                encoded_signal, self.feat_lens
            )

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs. We here
        do multi-task learning and the loss is a weighted sum of the ctc + seq2seq
        costs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        # Compute sequence loss against targets with EOS
        # tokens_eos, tokens_eos_lens = self.prepare_tokens(
        #     stage, batch.tokens_eos
        # )
        loss = self.compute_loss(
            predictions, 
            batch, 
            stage=stage, 
            reduction="mean", 
            weight=self.hparams.ctc_weight
        )

        if stage != sb.Stage.TRAIN:
            # Converted predicted tokens from indexes to words
            # predicted_words = [
            #     self.hparams.tokenizer.decode_ids(prediction).split(" ")
            #     for prediction in predictions["tokens"]
            # ]
            predicted_tokens = [h[0] for h in predictions['tokens']]
            self._valid_test_objectives(batch, predicted_tokens, stage)

        return loss
    
    def on_valid_test_stage_start(self, stage):
        # The on_stage_start method for the VALID and TEST stages.
        # This needs to be different because on the TRAIN stage we also use curriculum.
        # An example implementation is to initialize the cer, wer computers.
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN or (self.sorting in getattr(self.train_set, 'METRIC_SORTERS', [])):
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        super().on_stage_end(stage, stage_loss, epoch)
        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
                num_to_keep=getattr(self.hparams, "ckpts_to_keep", 1)
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
            if hasattr(self.hparams, 'cer_file'):
                with open(self.hparams.cer_file, "w") as c:
                    self.cer_metric.write_stats(c)

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        if getattr(self.hparams, "avg_ckpts", 1) > 1:
            ckpts = self.checkpointer.find_checkpoints(
                    max_key=max_key,
                    min_key=min_key,
                    max_num_checkpoints=self.hparams.avg_ckpts
            )
            model_state_dict = sb.utils.checkpoints.average_checkpoints(
                    ckpts, "model" 
            )
            self.hparams.model.load_state_dict(model_state_dict)
            self.checkpointer.save_checkpoint(name=f"AVERAGED-{self.hparams.avg_ckpts}")