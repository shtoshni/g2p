"""Seq2Seq model class that creates the computation graph.

Author: Trang Tran and Shubham Toshniwal
Contact: ttmt001@uw.edu, shtoshni@ttic.edu
Date: April, 2017
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops

import data_utils
from encoder import Encoder
from simple_decoder import SimpleDecoder


class Seq2SeqModel(object):
    """Implements the Encoder-Decoder model."""

    def __init__(self, buckets, isTraining, max_gradient_norm, batch_size,
                 learning_rate, learning_rate_decay_factor,
                 encoder_attribs, decoder_attribs):
        """Initializer of class that defines the computational graph.

        Args:
            buckets: List of input-output sizes that limit the amount of
                sequence padding (http://goo.gl/d8ybpl).
            isTraining: boolean that denotes training v/s evaluation.
            max_gradient_norm: Maximum value of gradient norm.
            batch_size: Minibatch size used for doing SGD.
            learning_rate: Initial learning rate of optimizer
            learning_rate_decay_factor: Multiplicative learning rate decay
                factor
            {encoder, decoder}_attribs: Dictionary containing attributes for
                {encoder, decoder} RNN.
        """
        self.buckets = buckets
        self.isTraining = isTraining
        self.batch_size = batch_size

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)

        # Number of gradient updates performed
        self.global_step = tf.Variable(0, trainable=False)
        # Number of epochs done
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)

        # Placeholder for encoder input IDs - Shape TxB
        self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                             name='encoder')
        _batch_size = self.encoder_inputs.get_shape()[1].value
        # Input sequence length placeholder
        self.seq_len = tf.placeholder(tf.int64, shape=[_batch_size],
                                      name="seq_len")
        # Output sequence length placeholder
        self.seq_len_target = tf.placeholder(tf.int64, shape=[_batch_size],
                                             name="seq_len_target")

        # Input to decoder RNN. This input has an initial extra symbol - GO -
        # that initiates the decoding process.
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                             name="decoder")
        # Targets are decoder inputs shifted by one thus, ignoring GO symbol
        self.targets = tf.slice(self.decoder_inputs, [1, 0], [-1, -1])

        # Initialize the encoder and decoder RNNs
        self.encoder = Encoder(isTraining, **encoder_attribs)
        self.decoder = SimpleDecoder(isTraining, **decoder_attribs)

        # First encode input
        self.encoder_hidden_states, self.final_state = \
            self.encoder.encode_input(self.encoder_inputs, self.seq_len)
        # Then decode
        self.outputs = \
            self.decoder.decode(self.decoder_inputs, self.seq_len_target,
                                self.encoder_hidden_states, self.final_state,
                                self.seq_len)
        # Training outputs and losses.
        self.losses = self.seq2seq_loss(self.outputs, self.targets,
                                        self.seq_len_target)

        if isTraining:
            # Gradients and parameter updation for training the model.
            params = tf.trainable_variables()
            print ("\nModel parameters:\n")
            for var in params:
                print (("{0}: {1}").format(var.name, var.get_shape()))
            print
            # Initialize optimizer
            opt = tf.train.AdamOptimizer(self.learning_rate)
            # Get gradients from loss
            gradients = tf.gradients(self.losses, params)
            # Clip the gradients to avoid the problem of gradient explosion
            # possible early in training
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norms = norm
            # Apply gradients
            self.updates = opt.apply_gradients(zip(clipped_gradients, params),
                                               global_step=self.global_step)

        # Model saver function
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    @staticmethod
    def seq2seq_loss(logits, targets, seq_len_target):
        """Calculate the cross entropy loss w.r.t. given target.

        Args:
            logits: A 2-d tensor of shape (TxB)x|V| containing the logit score
                per output symbol.
            targets: 2-d tensor of shape TxB that contains the ground truth
                output symbols.
            seq_len_target: Sequence length of output sequences. Required to
                mask padding symbols in output sequences.
        """
        with ops.name_scope("sequence_loss", [logits, targets]):
            flat_targets = tf.reshape(targets, [-1])
            cost = nn_ops.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=flat_targets)

            # Mask this cost since the output sequence is padded
            batch_major_mask = tf.sequence_mask(seq_len_target,
                                                dtype=tf.float32)
            time_major_mask = tf.transpose(batch_major_mask, [1, 0])
            weights = tf.reshape(time_major_mask, [-1])
            mask_cost = weights * cost

            loss = tf.reshape(mask_cost, tf.shape(targets))
            # Average the loss for each example by the # of timesteps
            cost_per_example = tf.reduce_sum(loss, reduction_indices=0) /\
                tf.cast(seq_len_target, tf.float32)
            # Return the average cost over all examples
            return tf.reduce_mean(cost_per_example)

    def step(self, sess, encoder_inputs, seq_len, decoder_inputs,
             seq_len_target):
        """Perform 1 minibatch update/evaluation.

        Args:
            sess: Tensorflow session where computation graph is created
            encoder_inputs: List of a minibatch of input IDs
            seq_len: Input sequence length
            decoder_inputs: List of a minibatch of output IDs
            seq_len_target: Output sequence length
        Returns:
            Output of a minibatch updated. The exact output depends on
            whether the model is in training mode or evaluation mode.
        """
        # Pass inputs via feed dict method
        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.decoder_inputs.name] = decoder_inputs
        input_feed[self.seq_len.name] = seq_len
        input_feed[self.seq_len_target.name] = seq_len_target

        if self.isTraining:
            # Important to have gradient updates as this operation is what
            # actually updates the parameters.
            output_feed = [self.updates,  self.gradient_norms, self.losses]
        else:
            # Evaluation
            output_feed = [self.outputs]

        outputs = sess.run(output_feed, input_feed)
        if self.isTraining:
            return outputs[1], outputs[2]
        else:
            return outputs[0]

    def get_batch(self, data, bucket_id=None):
        """Prepare minibatch from given data.

        Args:
            data: A list of datapoints (all from same bucket).
            bucket_id: Bucket ID of data. This is irrevelant for training but
                for evaluation we can limit the padding by the bucket size.
        Returns:
            Batched input IDs, input sequence length, output IDs & output
            sequence length
        """
        if not self.isTraining:
            # During evaluation the bucket size limits the amount of padding
            _, decoder_size = self.buckets[bucket_id]

        encoder_inputs, decoder_inputs = [], []
        batch_size = len(data)

        seq_len = np.zeros((batch_size), dtype=np.int64)
        seq_len_target = np.zeros((batch_size), dtype=np.int64)

        for i, sample in enumerate(data):
            encoder_input, decoder_input = sample
            seq_len[i] = len(encoder_input)
            if not self.isTraining:
                seq_len_target[i] = decoder_size
            else:
                # 1 is added to output sequence length because the EOS token is
                # crucial to "halt" the decoder. Consider it the punctuation
                # mark of a English sentence. Both are necessary.
                seq_len_target[i] = len(decoder_input) + 1

        # Maximum input and output length which limit the padding till them
        max_len_source = max(seq_len)
        max_len_target = max(seq_len_target)

        for i, sample in enumerate(data):
            encoder_input, decoder_input = sample
            # Encoder inputs are padded and then reversed.
            encoder_pad_size = max_len_source - len(encoder_input)
            encoder_pad = [data_utils.PAD_ID] * encoder_pad_size
            # Encoder input is reversed - https://arxiv.org/abs/1409.3215
            encoder_inputs.append(list(reversed(encoder_input)) + encoder_pad)

            # 1 is added to decoder_input because GO_ID is considered a part of
            # decoder input. While EOS_ID is also added, it's really used by
            # the target tensor (self.tensor) in the core code above.
            decoder_pad_size = max_len_target - (len(decoder_input) + 1)
            decoder_inputs.append([data_utils.GO_ID] +
                                  decoder_input +
                                  [data_utils.EOS_ID] +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Both the id sequences are made time major via transpose
        encoder_inputs = np.asarray(encoder_inputs, dtype=np.int32).T
        decoder_inputs = np.asarray(decoder_inputs, dtype=np.int32).T

        return encoder_inputs, seq_len, decoder_inputs, seq_len_target
