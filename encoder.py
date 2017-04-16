"""Encoder class of the seq2seq model.

Author: Shubham Toshniwal
Contact: shtoshni@ttic.edu
Date: March, 2017
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import embedding_ops

from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell
from tensorflow.python.ops import variable_scope


class Encoder(object):
    """Encoder class that encodes input sequence."""

    def __init__(self, isTraining, **attribs):
        """The initializer for encoder class.

        Args:
            isTraining: Whether the network is in training mode or not. This
                would affect whether dropout is used or not.
            **attribs: A dictionary of attributes used by encoder like:
                hidden_size: Hidden size of LSTM cell used for encoding
                num_layers: Number of hidden layers used
                bi_dir: Boolean determining whether the encoder is
                    bidirectional or not
                num_encoder_symbols: Vocabulary size of input symbols
                embedding_size: Embedding size used to feed in input symbols
                out_prob(Optional): (1 - Dropout probability)
        """
        self.isTraining = isTraining
        if self.isTraining:
            # Dropout is only used during training
            self.out_prob = attribs['out_prob']
        self.hidden_size = attribs['hidden_size']
        self.num_layers = attribs['num_layers']
        self.bi_dir = attribs['bi_dir']
        # Create the LSTM cell using the hidden size attribute
        self.cell = rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        if self.isTraining:
            # During training a dropout wrapper is used
            self.cell = rnn_cell.DropoutWrapper(self.cell,
                                                output_keep_prob=self.out_prob)

        self.vocab_size = attribs['num_encoder_symbols']
        self.emb_size = attribs['embedding_size']

    def _layer_encoder_input(self, encoder_inputs, seq_len, layer_depth=1):
        """Run a single LSTM on given input.

        Args:
            encoder_inputs: A 3-D Tensor input of shape TxBxE on which to run
                LSTM where T is number of timesteps, B is batch size and E is
                input dimension at each timestep.
            seq_len: A 1-D tensor that contains the actual length of each
                input in the batch. This ensures pad symbols are not
                processed as input.
            layer_depth: A integer denoting the depth at which the current
                layer is constructed. This information is necessary to
                differentiate the parameters of different layers.
        Returns:
            encoder_outputs: Output of LSTM, a 3-D tensor of shape TxBxH.
            final_state: Final hidden state of LSTM.
        """
        with variable_scope.variable_scope("RNNLayer%d" % (layer_depth),
                                           initializer=tf.random_uniform_initializer(-0.075, 0.075)):
            # Check if the encoder needs to be bidirectional or not.
            if self.bi_dir:
                (encoder_output_fw, encoder_output_bw), (final_state_fw, _) = \
                    rnn.bidirectional_dynamic_rnn(self.cell, self.cell,
                                                  encoder_inputs,
                                                  sequence_length=seq_len,
                                                  dtype=tf.float32,
                                                  time_major=True)
                # Concatenate the output of forward and backward layer
                encoder_outputs = tf.concat([encoder_output_fw,
                                             encoder_output_bw], 2)
                # Assume the final state is simply the final state of forward
                # layer. A combination of hidden states can also be done.
                final_state = final_state_fw
            else:
                encoder_outputs, final_state = \
                    rnn.dynamic_rnn(self.cell, encoder_inputs,
                                    sequence_length=seq_len, dtype=tf.float32,
                                    time_major=True)

            return encoder_outputs, final_state

    def encode_input(self, encoder_inp, seq_len):
        """Run the encoder on gives input.

        Args:
            encoder_inp: Input IDs that are time major i.e. TxB. These IDs are
                first passed through embedding layer before feeding to first
                LSTM layer.
            seq_len: Actual length of input time sequences.
        Returns:
            attention_states: Final encoder output for every input timestep.
                This tensor is used by attention-enabled decoders.
            final_state: Final state of encoder LSTM
        """
        with variable_scope.variable_scope("encoder"):
            embedding = variable_scope.get_variable(
                "embedding", [self.vocab_size, self.emb_size],
                initializer=tf.random_uniform_initializer(-1.0, 1.0))
            # Input ids are first embedded via the embedding lookup operation
            encoder_inputs = embedding_ops.embedding_lookup(embedding,
                                                            encoder_inp)

            final_states = []
            for layer_depth in xrange(self.num_layers):
                encoder_outputs, layer_final_state = \
                    self._layer_encoder_input(encoder_inputs, seq_len,
                                              layer_depth)
                # Output of previous layer is made input of next layer
                encoder_inputs = encoder_outputs
                final_states.append(layer_final_state)

            if self.num_layers == 1:
                final_state = final_states[0]
            else:
                # This is required to match the format used by MultiRNNCell
                final_state = tuple(final_states)

        # Make the output batch major for use by attention-enabled decoder
        attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
        return attention_states, final_state
