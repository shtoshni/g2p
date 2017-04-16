from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope

from decoder import Decoder


class SimpleDecoder(Decoder):
    """Implements the basic decoder of encoder-decoder framework."""

    def __init__(self, isTraining, **attribs):
        """Initializer that simply calls the base class initializer."""
        super(SimpleDecoder, self).__init__(isTraining, **attribs)

    def decode(self, decoder_inp, seq_len,
               encoder_hidden_states, final_state, seq_len_inp):
        """Basic decoder using LSTM to model output sequence."""
        # First prepare the decoder input - Embed the input and obtain the
        # relevant loop function
        decoder_inputs, loop_function = self.prepare_decoder_input(decoder_inp)

        # TensorArray is used to do dynamic looping over decoder input
        inputs_ta = tf.TensorArray(size=self.max_output, dtype=tf.float32)
        inputs_ta = inputs_ta.unstack(decoder_inputs)

        batch_size = tf.shape(decoder_inputs)[1]
        emb_size = decoder_inputs.get_shape()[2].value

        with variable_scope.variable_scope("rnn_decoder"):
            def simple_loop_function(time, cell_output, state, loop_state):
                # Check which sequences are processed
                elements_finished = (time >= tf.cast(seq_len, tf.int32))
                # finished would indicate if all output sequences have been
                # processed
                finished = tf.reduce_all(elements_finished)
                if cell_output is None:
                    # 0th time step. Initialize the decoder hidden state with
                    # final hidden state of encoder.
                    next_state = final_state
                    # Read the <GO> tag to start decoding
                    next_input = inputs_ta.read(time)
                    output = None
                else:
                    next_state = state
                    output = cell_output
                    if self.isTraining:
                        if loop_function is not None:
                            # Perform Scheduled sampling
                            # https://arxiv.org/abs/1506.03099
                            random_prob = tf.random_uniform([])
                            next_input = tf.cond(
                                finished,
                                lambda: tf.zeros([batch_size, emb_size],
                                                 dtype=tf.float32),
                                lambda: tf.cond(tf.greater_equal(
                                    random_prob, self.samp_prob),
                                    lambda: inputs_ta.read(time),
                                    lambda: loop_function(output))
                                    )
                        else:
                            # Read the decoder input till all output
                            # sequences are not finished.
                            next_input = tf.cond(
                                finished,
                                lambda: tf.zeros([batch_size, emb_size],
                                                 dtype=tf.float32),
                                lambda: inputs_ta.read(time)
                                )
                    else:
                        # During evaluation, the output of previous time step
                        # is fed into next time step
                        next_input = loop_function(output)
                return (elements_finished, next_input, next_state, output,
                        None)

        # outputs is a TensorArray with T=max(sequence_length) entries
        # of shape Bx|V|
        outputs, state, _ = rnn.raw_rnn(self.cell, simple_loop_function)
        # Concatenate the output across timesteps to get a tensor of TxBx|v|
        # shape
        outputs = outputs.concat()
        return outputs
