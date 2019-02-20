#!/usr/bin/env python

"""Process data to create vocabulary, bucketing etc.

Author: Shubham Toshniwal
Contact: shtoshni@ttic.edu
Date: March, 2017
"""

import os
import sys
import random
import cPickle as pickle
import argparse
from collections import OrderedDict as odict

# Special vocabulary symbols
_PAD = 'PAD'  # NULL symbol to pad sequences to same length in a minibatch
_GO = 'GO'    # Symbol denoting start of decoding
_EOS = 'EOS'  # Symbol denoting end of decoding
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2

# Buckets are useful to limit the amount of padding required since we use
# minibatch processing. For more detail refer this: http://goo.gl/d8ybpl
# Bucket sizes are specified as a list with each entry of the form -
# (Max input sequence length, Max output sequence length)
# Since this project is about Grapheme-to-Phoneme conversion, where the input
# sequence is characters in a word and output sequence is phonemes in word
# pronunciation, we use a single bucket to merely denote the max word length
# and max pronunciation length
_buckets = [(35, 35)]
FLAGS = object()


def parse_options():
    """Parse command line options."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_dir", "--data_dir",
                        default="../data/",
                        type=str, help="Data directory")
    parser.add_argument("-train_file", "--train_file",
                        default="train.txt", type=str,
                        help="Raw train data file")
    parser.add_argument("-dev_file", "--dev_file",
                        default="dev.txt", type=str,
                        help="Raw development data file")
    parser.add_argument("-test_file", "--test_file",
                        default="test.txt", type=str,
                        help="Raw test data file")

    args = parser.parse_args()
    return args


def read_data_line(line):
    """Process a line from data file.

    Assumes the following format:
    WORD    W ER D
    i.e. word tab separated with pronunciation where pronunciation has space
    separated phonemes

    Args:
        line: string representing the one line of the data file
    Returns:
        chars: list of characters
        phones: list of phones
    """
    line = line.strip()
    word, pronunciation = line.split("\t")

    chars = list(word.strip())
    phones = pronunciation.strip().split(" ")

    return chars, phones


def bucket_data(data, eval_data=False):
    """Perform bucketing on data."""
    bucketed_data = [[] for _ in _buckets]
    for input_ids, output_ids in data:
        for bucket_id, (max_inp_size, max_out_size) in enumerate(_buckets):
            # Refer the input and output size for bucketing training data
            # For evaluation data, bucketing is based solely on input size
            if len(input_ids) <= max_inp_size and \
                    (eval_data or len(output_ids) <= max_out_size):
                bucketed_data[bucket_id].append((input_ids, output_ids))
                break

    return bucketed_data


def read_and_bucket_data(data_file, eval_data=False):
    """Read data from file and return bucekted data."""
    try:
        data = pickle.load(open(data_file))
        bucketed_data = bucket_data(data, eval_data=eval_data)
        return bucketed_data
    except IOError as e:
        print ("I/O error({0}): {1}".format(e.errno, e.strerror))
        sys.exit(1)
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        sys.exit(1)


def batch_bucketed_data(bucketed_data, batch_size=64, shuffle=True):
    """Create batches in already bucketed data."""
    DEFAULT_BATCH_SIZE = 64
    if batch_size <= 0:
        print ("Assuming a batch size of %d since specified value is <= 0"
               % (DEFAULT_BATCH_SIZE))
        batch_size = DEFAULT_BATCH_SIZE

    batched_data = []
    for cur_bucket_data in bucketed_data:
        if shuffle:
            # Shuffle data in each bucket
            random.shuffle(cur_bucket_data)
        batched_data += [cur_bucket_data[i:i + batch_size]
                         for i in range(0, len(cur_bucket_data), batch_size)]
    if shuffle:
        # Shuffle the created batches
        random.shuffle(batched_data)
    return batched_data


def create_vocabulary(train_file):
    """Create input & output vocabulary from training data.

    Args:
        train_file: path to the training file
    Returns:
        vocab_{input|output}: dictionary mapping {input|output} to ID.
            Empty in case of no training data
    """
    vocab_input = odict()
    vocab_output = odict()

    input_count = 0
    output_count = 0
    try:
        if os.path.isfile(train_file):
            with open(train_file, 'r') as f_data:
                # Add the special symbols defined above
                for term in _START_VOCAB:
                    vocab_input[term] = input_count
                    vocab_output[term] = output_count
                    input_count += 1
                    output_count += 1

                for line in f_data:
                    input_list, output_list = read_data_line(line)
                    for input_item in input_list:
                        if not (input_item in vocab_input):
                            vocab_input[input_item] = input_count
                            input_count += 1

                    for output_item in output_list:
                        if not (output_item in vocab_output):
                            vocab_output[output_item] = output_count
                            output_count += 1
            return vocab_input, vocab_output
        else:
            raise ValueError("Training file %s not found." % train_file)
    except ValueError as e:
        print (e)
        sys.exit(1)
    except:
        print ("Error while creating vocabulary:", sys.exc_info()[0])
        sys.exit(1)


def write_vocabulary(vocab_dict, vocab_file):
    """Write vocabulary items to vocabulary file."""
    try:
        print ("Writing vocabulary file: %s" % vocab_file)
        with open(vocab_file, 'w') as f_vocab:
            for vocab_item in vocab_dict:
                f_vocab.write(vocab_item + "\n")
    except IOError as e:
        print ("IOError while writing vocabulary({0}): {1}".
               format(e.errno, e.strerror))
        sys.exit(1)


def initialize_vocabulary(vocabulary_path):
    """Load vocabulary from given file.

    Args:
        vocabulary_path: Path of vocabulary file.
    Returns:
        vocab: A dictionary mapping elements of vocabulary to index
        rev_vocab: A list mapping index to element of vocabulary
    """
    if os.path.isfile(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, 'r') as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)
        return {}, []


def process_data(data_file, data_split, input_vocab, output_vocab):
    """Convert the data provided into id's and dumps it as a pickle.

    Args:
        data_file - split of the data which is to be procesed e.g. train, test
        input_vocab - dictionary mapping input items to IDs
        output_vocab - dictionary mapping output items to IDs
    """
    proc_file = os.path.join(FLAGS.data_dir, data_split + ".pkl")
    data_file = os.path.join(FLAGS.data_dir, data_file)

    print ("Processing data file: %s" % data_file)
    proc_data = []
    try:
        if os.path.isfile(data_file):
            with open(data_file, 'r') as data_f:
                for line in data_f:
                    line = line.strip()
                    input_seq, output_seq = read_data_line(line)
                    try:
                        input_ids = [input_vocab[input_item]
                                     for input_item in input_seq]
                    except KeyError:
                        print ("Character not found in vocabulary in the\
                               following list of characters")
                        print (input_seq)
                        sys.exit(1)
                    try:
                        output_ids = [output_vocab[output_item]
                                      for output_item in output_seq]
                    except KeyError:
                        print ("Phone not found in vocabulary among the\
                               following of phones")
                        print (output_seq)
                        sys.exit(1)
                    proc_data.append((input_ids, output_ids))
        else:
            print ("Data file %s not found." % data_file)
            sys.exit(1)
    except:
        print ("Error processing file: %s" % data_file)
        sys.exit(1)
    try:
        print ("Writing processed data to file: %s" % proc_file)
        pickle.dump(proc_data, open(proc_file, "w"))
    except:
        print ("Error while saving the processed pickle file:", proc_file)


if __name__ == "__main__":
    FLAGS = parse_options()
    vocab_input_path = os.path.join(FLAGS.data_dir, "vocab.char")
    vocab_output_path = os.path.join(FLAGS.data_dir, "vocab.phone")

    # Create vocabularies by reading in the training data
    train_data_path = os.path.join(FLAGS.data_dir, FLAGS.train_file)
    input_vocab, output_vocab = create_vocabulary(train_data_path)

    write_vocabulary(input_vocab, vocab_input_path)
    write_vocabulary(output_vocab, vocab_output_path)

    for (data_file, data_split) in [(FLAGS.train_file, "train"),
                                    (FLAGS.dev_file, "dev"),
                                    (FLAGS.test_file, "test")]:
        process_data(data_file, data_split, input_vocab, output_vocab)
