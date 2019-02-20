"""Base decoder class of seq2seq model. Doesn't implement actual decoding.

Author: Shubham Toshniwal
Contact: shtoshni@ttic.edu
Date: March, 2017
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops

from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell
from tensorflow.python.ops import variable_scope



class Decoder(object):
    """Base class for decoder in encoder-decoder framework."""

    def __init__(self, isTraining, **attribs):
        """The initializer for decoder class.

        Args:
            isTraining: Whether the network is in training mode or not. This
                would affect whether dropout and sampling are used or not.
            **attribs: A dictionary of attributes used by encoder like:
                hidden_size: Hidden size of LSTM cell used for decoding
                num_layers: Number of hidden layers used
                num_decoder_symbols: Vocabulary size of output symbols
                embedding_size: Embedding size used to feed in input symbols
                out_prob (Optional): (1 - Dropout probability)
                samp_prob (Optional): Sampling probability for sampling output
                    of previous step instead of using ground truth during
                    training.
                max_output (Optional): Maximum length of output sequence.
                    Assumed to be 100 if not specified.
        """
        self.isTraining = isTraining
        if self.isTraining:
            self.out_prob = attribs['out_prob']
            self.isSampling = False
            if ("samp_prob" in attribs) and attribs["samp_prob"] > 0.0:
                self.isSampling = True
                self.samp_prob = attribs['samp_prob']

        self.hidden_size = attribs['hidden_size']
        self.num_layers = attribs['num_layers']
        self.vocab_size = attribs['num_decoder_symbols']
        self.cell = self.set_cell_config()

        self.emb_size = attribs['embedding_size']

        self.max_output = 100   # Maximum length of output
        if 'max_output' in attribs:
            self.max_output = attribs['max_output']

    def set_cell_config(self):
        """Create the LSTM cell used by decoder."""
        # Use the BasicLSTMCell - https://arxiv.org/pdf/1409.2329.pdf
        cell = rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        if self.isTraining:
            # During training we use a dropout wrapper
            cell = rnn_cell.DropoutWrapper(cell,
                                           output_keep_prob=self.out_prob)
        if self.num_layers > 1:
            # If RNN is stacked then we use MultiRNNCell class
            cell = rnn_cell.MultiRNNCell([cell] * self.num_layers,
                                         state_is_tuple=True)

        # Use the OutputProjectionWrapper to project cell output to output
        # vocab size. This projection is fine for a small vocabulary output
        # but would be bad for large vocabulary output spaces.
        cell = rnn_cell.OutputProjectionWrapper(cell, self.vocab_size)
        return cell

    def prepare_decoder_input(self, decoder_inputs):
        """Do this step before starting decoding.

        This step converts the decoder IDs to vectors and
        Args:
            decoder_inputs: Time major decoder IDs
        Returns:
            embedded_inp: Embedded decoder input.
            loop_function: Function for getting next timestep input.
        """
        with variable_scope.variable_scope("decoder"):
            # Create an embedding matrix
            embedding = variable_scope.get_variable(
                "embedding", [self.vocab_size, self.emb_size],
                initializer=tf.random_uniform_initializer(-1.0, 1.0))
            # Embed the decoder input via embedding lookup operation
            embedded_inp = embedding_ops.embedding_lookup(embedding,
                                                          decoder_inputs)

        if self.isTraining:
            if self.isSampling:
                # This loop function samples the output from the posterior
                # and embeds this output.
                loop_function = self._sample_argmax(embedding)
            else:
                loop_function = None
        else:
            # Get the loop function that would embed the maximum posterior
            # symbol. This funtion is used during decoding in RNNs
            loop_function = self._get_argmax(embedding)

        return (embedded_inp, loop_function)

    @abstractmethod
    def decode(self, decoder_inp, seq_len,
               encoder_hidden_states, final_state, seq_len_inp):
        """Abstract method that needs to be extended by Inheritor classes.

        Args:
            decoder_inp: Time major decoder IDs, TxB that contain ground truth
                during training and are dummy value holders at test time.
            seq_len: Output sequence length for each input in minibatch.
                Useful to limit the computation to the max output length in
                a minibatch.
            encoder_hidden_states: Batch major output, BxTxH of encoder RNN.
                Useful with attention-enabled decoders.
            final_state: Final hidden state of encoder RNN. Useful for
                initializing decoder RNN.
            seq_len_inp: Useful with attention-enabled decoders to mask the
                outputs corresponding to padding symbols.
        Returns:
            outputs: Time major output, TxBx|V|, of decoder RNN.
        """
        pass

    def _get_argmax(self, embedding):
        """Return a function that returns the previous output with max prob.

        Args:
            embedding : Embedding matrix for embedding the symbol
        Returns:
            loop_function: A function that returns the embedded output symbol
                with maximum probability (logit score).
        """
        def loop_function(logits):
            max_symb = math_ops.argmax(logits, 1)
            emb_symb = embedding_ops.embedding_lookup(embedding, max_symb)
            return emb_symb

        return loop_function

    def _sample_argmax(self, embedding):
        """Return a function that samples from posterior over previous output.

        Args:
            embedding : Embedding matrix for embedding the symbol
        Returns:
            loop_function: A function that samples the output symbol from
            posterior and embeds the sampled symbol.
        """
        def loop_function(prev):
            """The closure function returned by outer function.

            Args:
                prev: logit score for previous step output
            Returns:
                emb_prev: The embedding of output symbol sampled from
                    posterior over previous output.
            """
            # tf.multinomial performs sampling given the logit scores
            # Reshaping is required to remove the extra dimension introduced
            # by sampling for a batch size of 1.
            prev_symbol = tf.reshape(tf.multinomial(prev, 1), [-1])
            emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
            return emb_prev

        return loop_function
