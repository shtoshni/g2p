#!/usr/bin/env python

"""Main file that manages training and evaluation for G2P.

Author: Shubham Toshniwal
Contact: shtoshni@ttic.edu
Date: March, 2017
Publication: https://arxiv.org/abs/1610.06540 (IEEE SLT 2017)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Hides verbose log messages

import sys
import time
import argparse
import operator
from bunch import bunchify
from copy import deepcopy

import numpy as np
import editdistance as ed
import data_utils
import seq2seq_model

import tensorflow as tf

FLAGS = object()
_buckets = data_utils._buckets  # Import bucket sizes from data_utils


def parse_options():
    """Parse command line options."""
    parser = argparse.ArgumentParser()

    # Gradient calculation parameters
    parser.add_argument("-lr", "--learning_rate",
                        default=1e-3, type=float,
                        help="learning rate")
    parser.add_argument("-lr_decay", "--learning_rate_decay_factor",
                        default=0.8, type=float,
                        help="multiplicative decay factor for learning rate")
    parser.add_argument("-max_gnorm", "--max_gradient_norm",
                        default=5.0, type=float,
                        help="Maximum allowed norm of gradients")
    parser.add_argument("-bsize", "--batch_size",
                        default=256, type=int,
                        help="Minibatch Size")

    # Common Encoder and Decoder parameters
    parser.add_argument("-esize", "--embedding_size",
                        default=30, type=int,
                        help="Input Embedding Size")
    parser.add_argument("-hsize", "--hidden_size",
                        default=256, type=int,
                        help="Hidden layer size")
    parser.add_argument("-num_layers", "--num_layers",
                        default=3, type=int,
                        help="Number of stacked layers in enc-dec RNNs")
    parser.add_argument("-out_prob", "--output_keep_prob",
                        default=0.8, type=float,
                        help="Output keep probability for dropout")

    # Separate encoder parameters
    parser.add_argument("-bi", "--bi_dir",
                        default=False, action='store_true',
                        help="Whether the encoder is bi-directional or not")

    # Separate decoder parameters
    parser.add_argument("-samp_prob", "--sampling_probability",
                        default=0.0, type=float,
                        help="Scheduled sampling probability")

    # Data parameters
    parser.add_argument("-data_dir", "--data_dir",
                        default="../data",
                        type=str, help="Data directory")
    parser.add_argument("-sv_file", "--source_vocab_file",
                        default="vocab.char", type=str,
                        help="Vocabulary file for characters")
    parser.add_argument("-tv_file", "--target_vocab_file",
                        default="vocab.phone", type=str,
                        help="Vocabulary file for phonemes")

    # Model storing parameters
    parser.add_argument("-tb_dir", "--train_base_dir",
                        default="../models",
                        type=str, help="Base training directory")
    parser.add_argument("-num_check", "--steps_per_checkpoint",
                        default=100, type=int,
                        help="Number of steps after which model ")
    parser.add_argument("-max_epochs", "--max_epochs",
                        default=100, type=int,
                        help="Maximum number of epochs")

    parser.add_argument("-eval", "--eval",
                        default=False, action="store_true",
                        help="Evaluate using the last saved model")
    parser.add_argument("-run_id", "--run_id",
                        default=0, type=int,
                        help="Run ID parameter to distinguish diff. runs")

    args = parser.parse_args()
    # Convert the parse arguments to dictionary for adding new entries
    arg_dict = vars(args)

    samp_string = ""
    if arg_dict["sampling_probability"] > 0.0:
        samp_string = "samp_" + str(arg_dict["sampling_probability"]) + "_"

    # Make a directory name that incorporates key model parameters in it's name
    train_dir = ('lr' + '_' + str(arg_dict['learning_rate']) + '_' +
                 'bsize' + '_' + str(arg_dict['batch_size']) + '_' +
                 'esize' + '_' + str(arg_dict['embedding_size']) + '_' +
                 'hsize' + '_' + str(arg_dict['hidden_size']) + '_' +
                 'num_layers' + '_' + str(arg_dict['num_layers']) + '_' +

                 'bi' + '_' + str(arg_dict['bi_dir']) + '_' +
                 'out_prob' + '_' + str(arg_dict['output_keep_prob']) + '_' +
                 samp_string +
                 'run_id' + '_' + str(arg_dict['run_id']) + '_' +
                 'g2p')

    try:
        if not os.path.exists(arg_dict['train_base_dir']):
            os.makedirs(arg_dict['train_base_dir'])
    except:
        print ("Could not create base folder %s that contains model "
               "directories for different runs" % arg_dict['train_base_dir'])
        sys.exit(1)
    arg_dict['train_dir'] = \
        os.path.join(arg_dict['train_base_dir'], train_dir)

    source_vocab_path = os.path.join(arg_dict['data_dir'],
                                     arg_dict['source_vocab_file'])
    target_vocab_path = os.path.join(arg_dict['data_dir'],
                                     arg_dict['target_vocab_file'])
    source_vocab, _ = data_utils.initialize_vocabulary(source_vocab_path)
    target_vocab, _ = data_utils.initialize_vocabulary(target_vocab_path)

    # Add the vocab size fields
    arg_dict['source_vocab_size'] = len(source_vocab)
    arg_dict['target_vocab_size'] = len(target_vocab)

    # common_attribs contains attribute values common to encoder and decoder
    # RNNs such as number of hidden units, number of layers etc
    common_attribs = {}
    common_attribs['out_prob'] = arg_dict['output_keep_prob']
    common_attribs['hidden_size'] = arg_dict['hidden_size']
    common_attribs['num_layers'] = arg_dict['num_layers']
    common_attribs['embedding_size'] = arg_dict['embedding_size']

    # Add the encoder-specific attributes to encoder_attribs
    encoder_attribs = deepcopy(common_attribs)
    encoder_attribs['num_encoder_symbols'] = arg_dict['source_vocab_size']
    encoder_attribs['bi_dir'] = arg_dict['bi_dir']
    arg_dict['encoder_attribs'] = encoder_attribs

    # Add the decoder-specific attributes to decoder_attribs
    decoder_attribs = deepcopy(common_attribs)
    decoder_attribs['samp_prob'] = arg_dict['sampling_probability']
    decoder_attribs['num_decoder_symbols'] = arg_dict['target_vocab_size']
    arg_dict['decoder_attribs'] = decoder_attribs

    if not arg_dict['eval']:
        # Create a model directory if one doesn't exist
        try:
            if not os.path.exists(arg_dict['train_dir']):
                os.makedirs(arg_dict['train_dir'])
        except:
            print ("Could not create model directory %s to store checkpoints"
                   % (arg_dict['train_dir']))
            sys.exit(1)
        # Sort the arg_dict to create a parameter file
        parameter_file = 'parameters.txt'
        sorted_args = sorted(arg_dict.items(), key=operator.itemgetter(0))

        with open(os.path.join(arg_dict['train_dir'], parameter_file), 'w') as g:
            for arg, arg_val in sorted_args:
                g.write(arg + "\t" + str(arg_val) + "\n")

    # Create an object from the argument dictionary and return
    options = bunchify(arg_dict)
    return options


def create_model_graph(session, isTraining):
    """Create the model graph by creating an instance of Seq2SeqModel."""
    return seq2seq_model.Seq2SeqModel(
        _buckets, isTraining, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        FLAGS.encoder_attribs, FLAGS.decoder_attribs)


def get_model(session, isTraining=True):
    """Create the model graph/Restore from a prev. checkpoint."""
    model = create_model_graph(session, isTraining=isTraining)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    steps_done = 0
    try:
        # Restore model parameters
        model.saver.restore(session, ckpt.model_checkpoint_path)
        sys.stdout.write("Reading model parameters from %s\n" %
                         ckpt.model_checkpoint_path)
        sys.stdout.flush()
        steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
        print("loaded from %d done steps" % (steps_done))
    except:
        sys.stdout.write("Created model with fresh parameters.\n")
        sys.stdout.flush()
        # Initialize model parameters
        session.run(tf.global_variables_initializer())
    return model, steps_done


def train():
    """Train the G2P model."""
    with tf.Session(config=
                    tf.ConfigProto(intra_op_parallelism_threads=2,
                                   inter_op_parallelism_threads=2)) as sess:
        with tf.variable_scope("model", reuse=None):
            # Creates the training graph
            model, steps_done = get_model(sess, isTraining=True)
        with tf.variable_scope("model", reuse=True):
            # Creates the validation graph that reuses training parameters
            mvalid = create_model_graph(sess, isTraining=False)

        print ("Models created")
        print ("Reading data from %s" % FLAGS.data_dir)
        sys.stdout.flush()

        # Load train and dev data
        train_data = data_utils.read_and_bucket_data(
            os.path.join(FLAGS.data_dir, "train.pkl"))
        dev_set = data_utils.read_and_bucket_data(
            os.path.join(FLAGS.data_dir, "dev.pkl"))

        step_time, loss = 0.0, 0.0
        previous_losses = []

        epoch_id = model.epoch.eval()

        val_wer_window = []
        window_size = 3

        if steps_done > 0:
            # The model saved would have wer and per better than 1.0
            best_wer, _ = calc_levenshtein_loss(mvalid, sess, dev_set)
        else:
            best_wer = 1.0

        print ("Starting training !!\n")
        sys.stdout.flush()
        while (epoch_id < FLAGS.max_epochs):
            steps = 0.0
            # Batch the data (Also shuffles the data)
            batch_data = data_utils.batch_bucketed_data(
                train_data, batch_size=FLAGS.batch_size)
            for batch in batch_data:
                # Run a minibatch update and record the run times
                start_time = time.time()
                encoder_inputs, seq_len, decoder_inputs, seq_len_target = \
                    model.get_batch(batch)
                _, step_loss = model.step(sess, encoder_inputs, seq_len,
                                          decoder_inputs, seq_len_target)

                step_time += (time.time() - start_time)
                loss += step_loss

                steps += 1.0

            # Increase the epoch counter
            epoch_id += 1
            sess.run(model.epoch_incr)

            step_time /= steps
            loss /= steps
            perplexity = np.exp(loss) if loss < 300 else float('inf')
            print ("Epoch %d global step %d learning rate %.4f step-time %.2f"
                   " perplexity %.4f" % (epoch_id, model.global_step.eval(),
                                         model.learning_rate.eval(), step_time,
                                         perplexity))
            if len(previous_losses) >= 3 and loss > max(previous_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            step_time, loss = 0.0, 0.0

            # Calculate validation result
            val_wer, val_per = calc_levenshtein_loss(mvalid, sess, dev_set)
            print("Validation WER: %.5f, PER: %.5f" % (val_wer, val_per))
            sys.stdout.flush()

            # Validation WER is a moving window, we add the new entry and pop
            # the oldest one
            val_wer_window.append(val_wer)
            if len(val_wer_window) > window_size:
                val_wer_window.pop(0)
            avg_wer = sum(val_wer_window)/float(len(val_wer_window))
            print("Average Validation WER %.5f" % (avg_wer))
            sys.stdout.flush()

            # The best model is decided based on average validation WER to
            # remove noisy cases of one off validation success
            if best_wer > avg_wer:
                # Save the best model
                best_wer = avg_wer
                print("Saving Updated Model")
                sys.stdout.flush()
                checkpoint_path = os.path.join(FLAGS.train_dir, "g2p.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step,
                                 write_meta_graph=False)
            print


def calc_levenshtein_loss(model, sess, eval_set):
    """Calculate the actual loss function for G2P.

    Args:
        model: Seq2SeqModel instance
        sess: Tensorflow session with the model compuation graph
        eval_set: Bucketed evaluation set
    Returns:
        wer: Word Error Rate
        per: Phoneme Error Rate
    """
    total_words = 0
    total_phonemes = 0
    wer = 0
    per = 0
    edit_distances = []

    for bucket_id in xrange(len(data_utils._buckets)):
        cur_data = eval_set[bucket_id]
        for batch_offset in xrange(0, len(cur_data), FLAGS.batch_size):
            batch = cur_data[batch_offset:batch_offset + FLAGS.batch_size]
            num_instances = len(batch)
            # Each instance is a pair of ([Input sequence], [Output sequence])
            inp_ids = [inst[0] for inst in batch]
            gt_ids = [inst[1] for inst in batch]
            encoder_inputs, seq_len, decoder_inputs, seq_len_target = \
                model.get_batch(batch, bucket_id=bucket_id)
            # Run the model to get output_logits of shape TxBx|V|
            output_logits = model.step(sess, encoder_inputs, seq_len,
                                       decoder_inputs, seq_len_target)

            # This is a greedy decoder and output is just argmax at each timestep
            outputs = np.argmax(output_logits, axis=1)
            # Reshape the output and make it batch major via transpose
            outputs = np.reshape(outputs, (max(seq_len_target), num_instances)).T
            for idx in xrange(num_instances):
                cur_output = list(outputs[idx])
                if data_utils.EOS_ID in cur_output:
                    cur_output = cur_output[:cur_output.index(data_utils.EOS_ID)]

                gt = gt_ids[idx]
                # Calculate the edit distance from ground truth
                distance = ed.eval(gt, cur_output)
                edit_distances.append((inp_ids[idx], distance, len(gt)))

    edit_distances.sort()

    # Aggregate the edit distances for each word
    word_to_edit = {}
    for edit_distance in edit_distances:
        word, distance, num_phonemes = edit_distance
        word = tuple(word)
        if word in word_to_edit:
            word_to_edit[word].append((distance, num_phonemes))
        else:
            word_to_edit[word] = [(distance, num_phonemes)]

    total_words = len(word_to_edit)
    for word in word_to_edit:
        # Pick the ground truth that's closest to output since their can be
        # multiple pronunciations
        distance, num_phonemes = min(word_to_edit[word])
        if distance != 0:
            wer += 1
            per += distance
        total_phonemes += num_phonemes

    try:
        wer = float(wer)/float(total_words)
    except ZeroDivisionError:
        print ("0 words in evaluation set")
        wer = 1.0
    try:
        per = float(per)/float(total_phonemes)
    except ZeroDivisionError:
        print ("0 phones in evaluation set")
        per = 1.0
    return wer, per


def evaluate():
    """Perform evaluation on dev/test data."""
    with tf.Session() as sess:
        # Load model
        with tf.variable_scope("model"):
            model, _ = get_model(sess, isTraining=False)
            test_set = data_utils.read_and_bucket_data(
                os.path.join(FLAGS.data_dir, "test.pkl"))
            wer, per = calc_levenshtein_loss(model, sess, test_set)
            print ('Total WER %.5f, PER %.5f' % (wer, per))


if __name__ == "__main__":
    FLAGS = parse_options()
    if FLAGS.eval:
        evaluate()
    else:
        train()
