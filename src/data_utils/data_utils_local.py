"""
Contains functions to split the data and generate the vocab
"""

import os
import random
import configparser
import glob
import copy
import json
import logging
import string
import shutil
from tqdm import tqdm
from collections import defaultdict

from nltk.util import ngrams
from collections import Counter
import numpy as np
import pandas as pd

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def data_split_pipeline(config):
    def _train_val_test_split(config):
        # Grab files from raw data dir
        files = glob.glob(config["data_path"]["raw_data_dir"] + "/*000.csv")

        # Shuffle and split
        train_size = float(config["dataset"]["train_size"])
        random.shuffle(files)
        n_samples = len(files)
        train_files = files[: int(train_size * n_samples)]
        val_test_files = files[int(train_size * n_samples) :]
        val_files = val_test_files[: (len(val_test_files) // 2)]
        test_files = val_test_files[(len(val_test_files) // 2) :]

        return train_files, val_files, test_files

    def _write_to_file(files, data_dir):
        for file in files:
            shutil.copy(file, data_dir)

    train, val, test = _train_val_test_split(config)
    _write_to_file(train, config["data_path"]["train_dir"])
    _write_to_file(val, config["data_path"]["val_dir"])
    _write_to_file(test, config["data_path"]["test_dir"])


def split_subtrain(config):
    files = glob.glob(config["data_path"]["train_dir"] + "/*000.csv")
    random.shuffle(files)
    files = files[:5]

    for file in files:
        shutil.copy(file, config["data_path"]["train_subset_dir"])


# def make_vocab(config):
#     files = glob.glob(config["data_path"]["raw_data_dir"] + "/*000.csv")
#     text_vocab = set(string.ascii_lowercase + string.ascii_uppercase + string.digits)
#     action_vocab = set()

#     for file in files:
#         sample = pd.read_csv(file, usecols=["change_text", "action_type"])
#         input_strs = sample["change_text"].unique()
#         actions = sample["action_type"].unique()
#         text_vocab.update(input_strs)
#         action_vocab.update(actions)

#     text_vocab_path = config["resource_path"]["text_vocab_path"]
#     with open(text_vocab_path, "w") as f:
#         for token in text_vocab:
#             token = repr(token).replace("'", "")
#             f.write(f"{token}\n")

#     action_vocab_path = config["resource_path"]["action_vocab_path"]
#     with open(action_vocab_path, "w") as f:
#         for token in action_vocab:
#             f.write(f"{token}\n")

#     return text_vocab, action_vocab


def make_vocab_mapping_vector(input_dir, text_mapping_out, action_mapping_out):
    token2id_text = {"<pad>": 0, "<unk>": 1, "<long>": 2, "<mask>": 3, '<sos>': 4, '<eos>': 5}
    token2id_action = copy.deepcopy(token2id_text)
    text_counter = len(token2id_text)
    action_counter = len(token2id_action)

    for i in range(127):
        character = chr(i).lower()
        if character not in token2id_text:
            token2id_text[character] = text_counter
            text_counter += 1

    data = _read_files(input_dir)
    pbar = tqdm(data.items())
    for sample_id, sample in pbar:
        for stroke in sample:
            action = stroke["status"]

            if action not in token2id_action:
                token2id_action[action] = action_counter
                action_counter += 1

    # with open(text_mapping_out, "w") as f:
    #     json.dump(token2id_text, f)
    with open(action_mapping_out, "w") as f:
        json.dump(token2id_action, f)


def make_vocab_mapping_token(input_dir):
    data = _read_files(input_dir)
    with open("src/resources/timing_bins.json", "r") as f:
        binnings = json.load(f)
    binning = binnings["0.85-7"]
    binning.insert(0, 0)
    binning.append(float("inf"))
    token2id = {"<pad>": 0, "<unk>": 1, "<long>": 2, "<mask>": 3}
    token_freq = defaultdict(int)
    counter = len(token2id)
    key_sequence = []

    pbar = tqdm(data.items())
    for sample_id, sample in pbar:
        for stroke in sample:
            text = (
                stroke["change_text"]
                if len(str(stroke["change_text"])) < 4
                else "<long>"
            )
            action = stroke["action_type"]
            keystroke = "^".join([str(text).lower(), str(action)])
            key_sequence.append(keystroke)
            token_freq[keystroke] += 1

    bigrams = ngrams(key_sequence, 2)
    bigram_freq = Counter(bigrams)
    bigram_freq = {str(key): value for key, value in bigram_freq.items()}

    trigrams = ngrams(key_sequence, 3)
    trigram_freq = Counter(trigrams)
    trigram_freq = {str(key): value for key, value in trigram_freq.items()}
    trigram_freq = dict(
        sorted(trigram_freq.items(), key=lambda item: item[1], reverse=True)
    )

    token_freq = dict(
        sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
    )

    for item, _ in token_freq.items():
        token2id[item] = counter
        counter += 1

    with open("src/resources/token2id_token_reduced.json", "w") as f:
        json.dump(token2id, f)
    with open("src/resources/token_freq_reduced.json", "w") as f:
        json.dump(token_freq, f)
    with open("src/resources/char_bigram_freq_reduced.json", "w") as f:
        json.dump(bigram_freq, f)
    with open("src/resources/char_trigram_freq_reduced.json", "w") as f:
        json.dump(trigram_freq, f)


def tokenize_dataset(input_dir, output_path):
    data = _read_files(input_dir)
    with open("src/resources/timing_bins.json", "r") as f:
        binnings = json.load(f)
    binning = binnings["0.85-7"]
    binning.insert(0, 0)
    binning.append(float("inf"))
    samples = []
    key_sequence = []

    pbar = tqdm(data.items())
    for sample_id, sample in pbar:
        keystroke_sequence = []
        absolute_position = 0

        for stroke in sample:
            relative_position = stroke["log_position"] - absolute_position
            absolute_position = stroke["log_position"]
            text = stroke["change_text"]
            action = stroke["action_type"]
            # interval_binning = np.digitize(stroke["interkey_interval"], binning)
            interval_binning = min(
                round(np.log(stroke['interkey_interval'] + 0.00001), 1), 7
            )
            keystroke = "^".join(
                [
                    str(text),
                    str(action),
                    str(interval_binning),
                    str(relative_position),
                ]
            )

            keystroke_sequence.append(keystroke)
            key_sequence.append(keystroke)
        samples.append({"sample_id": sample_id, "encoded_sequence": keystroke_sequence})

    # with open(output_path, "w") as f:
    #     json.dump(samples, f)

    trigrams = ngrams(key_sequence, 3)
    trigram_freq = Counter(trigrams)
    trigram_freq = {str(key): value for key, value in trigram_freq.items()}
    trigram_freq = dict(
        sorted(trigram_freq.items(), key=lambda item: item[1], reverse=True)
    )

    with open('src/resources/vector_trigram_freq.json', 'w') as f:
        json.dump(trigram_freq, f)

    return samples


def _read_files(input_path):
    """
    returns a dictionary of of sample_id : keylog_sequence
    keylog_sequence is a list of dictionaries, each element
    is information associated with one keystroke
    """

    out = {}
    cols = [
        "appointment_id",
        'status',
        "action_type",
        "log_action",
        "log_position",
        "change_text",
        "interkey_interval",
    ]
    files = glob.glob(input_path + "/*000.csv")
    pbar = tqdm(files)

    for file in pbar:
        pbar.set_description(f"reading from {file}")
        df = pd.read_csv(file, usecols=cols)
        sample_id = df["appointment_id"][0]
        sample = df.to_dict("records")

        out[sample_id] = sample

    return out


if __name__ == "__main__":
    config_path = os.path.join("configs", "config.ini")
    config = configparser.ConfigParser()
    config.read(config_path)
    random.seed(int(config["general"]["seed"]))

    # data_split_pipeline(config)
    # split_subtrain(config)
    # make_vocab(config)
    make_vocab_mapping_vector(
        "data/raw/all_data",
        "src/resources/token2id_vector_text.json",
        "src/resources/token2id_vector_action_reduced.json",
    )
    # make_vocab_mapping_token("data/raw/all_data")
    # tokenize_dataset('data/processed/train', 'data/processed/tokenized/train_subset.json')
