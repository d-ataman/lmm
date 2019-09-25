from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn
import numpy
import onmt
import onmt.io
import onmt.modules


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct.to(dtype=torch.float) / self.n_words.to(dtype=torch.float))

    def ppl(self):
        return math.exp(min(self.loss / self.n_words.float(), 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, epoch):
        t = self.elapsed_time()
        values = {
            "ppl": self.ppl(),
            "accuracy": self.accuracy(),
            "tgtper": self.n_words / t,
            "lr": lr,
        }
        writer.add_scalars(prefix, values, epoch)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            src_data_type(string): type of the source input: [words|trigrams|img|audio]
            tgt_data_type(string): type of the source input: [words|characters|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, src_data_type='words', tgt_data_type='words', 
                 norm_method="sents", grad_accum_count=1, batch_size=64):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.src_data_type = src_data_type
        self.tgt_data_type = tgt_data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.batch_size = batch_size

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0

        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1


        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                normalization += batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            total_stats.start_time, self.optim.lr,
                            report_stats)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        chars_len = 0

        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:

            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.src_data_type)
            if self.src_data_type == 'words':
                _, src_lengths = batch.src
            elif self.src_data_type in ['trigrams', 'characters']:
                src_lengths = batch.src[1]
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt', self.tgt_data_type)

            # F-prop through the model.
            if self.src_data_type in ['trigrams', 'characters'] or self.tgt_data_type == 'characters':

                outputs, attns, dec_state, logloss = \
                self.model(src, tgt, src_lengths, batch.batch_size)

                if self.tgt_data_type == 'characters':
                    # Transform batch to fit output
                    chars_len, batch_seqlen = batch.tgt.size()
                    seqlen = batch_seqlen // batch.batch_size
                    batch.tgt = torch.stack(batch.tgt.split(seqlen, dim=1), dim=2)
                    batch.tgt = batch.tgt[1:,1:,:]
                    batch.tgt = torch.cat(batch.tgt.unbind(1), dim=0)

            else:
                outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns, self.tgt_data_type, logloss)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):


        if self.grad_accum_count > 1:
            self.model.zero_grad()


        #counter = 0 #used for updating the loss until batch_size=64 is reached
        #loss_c = torch.zeros(1).long().cuda() 
        for batch in true_batchs:
            #counter = counter + 1

            dec_state = None
            
            target_size = batch.tgt.size(0)
            batch_size = batch.batch_size
            
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src = onmt.io.make_features(batch, 'src', self.src_data_type)
            if self.src_data_type == 'words':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            elif self.src_data_type in ['trigrams', 'characters']:
                src_lengths = batch.src[1]
                report_stats.n_src_words += src_lengths[1].sum()
            else:
                src_lengths = None

            tgt_outer = onmt.io.make_features(batch, 'tgt', self.tgt_data_type)

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()

                #If we are using trigrams or characters we need to pass the batch size as an additional parameter
                if self.src_data_type in ["trigrams", "characters"] or self.tgt_data_type == "characters":
                    outputs, attns, dec_state, logloss = \
                    self.model(src, tgt, src_lengths, batch.batch_size, dec_state)

                    if self.tgt_data_type == "characters":
                        # Transform batch to fit output
                        chars_len, batch_seqlen = batch.tgt.size()
                        seqlen = batch_seqlen // batch.batch_size
                        batch.tgt = torch.stack(batch.tgt.split(seqlen, dim=1), dim=2)
                        batch.tgt = batch.tgt[1:,1:,:] # remove BOS and BOWs
                        batch.tgt = torch.cat(batch.tgt.unbind(1), dim=0)
                        #batch.tgt = torch.stack(batch.tgt).permute(2,1,0)
                        #batch.tgt = batch.tgt.reshape(chars_len*(seqlen-1), batch.batch_size)
                        # Update trunc_size
                        trunc_size = batch.tgt.size(0)

                else:
                    outputs, attns, dec_state, _ = \
                    self.model(src, tgt, src_lengths, dec_state)


                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization, self.tgt_data_type, logloss)
                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()
