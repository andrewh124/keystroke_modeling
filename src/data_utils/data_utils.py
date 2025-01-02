import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import configparser
import glob
import json
import logging
import random
import copy
import typing as t
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _write_to_jsonl(obj_out: t.List, path_out: str) -> None:
    """
    Write file to jsonl
    Doesnt check for existing file
    """
    path_out = Path(path_out)
    with open(path_out, "w", encoding="utf-8") as f:
        for line in obj_out:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")


def _write_to_json(obj_out: t.Union[t.Dict, t.List], path_out: str) -> None:
    """
    Write file to json
    Doesnt check for existing file
    """
    path_out = Path(path_out)
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(obj_out, f, ensure_ascii=False, indent=2)


def get_essay_pairs(all_files_dir: str, output_mapping_path: str) -> t.List:
    """
    Each person writes 2 essays
    Finds the corresponding pair for each person
    Writes the mapping to a jsonl file
    Mapping {person: [essay_1, essay_2]}
    """
    all_files_dir = Path(all_files_dir)
    all_files = glob.glob(str(all_files_dir / "*.json"))
    mapping = defaultdict(list)

    for file in tqdm(all_files):
        file_name = str(Path(file).stem)
        person, _ = file_name.split("_")
        mapping[person].append(file)

    _write_to_json(mapping, output_mapping_path)
    logger.info(f"Dumped mapping to {output_mapping_path}")

    return mapping


def train_val_test_split(mapping_file: str, config: t.Type[configparser.ConfigParser]):
    """
    Read the json mapping for person - essay pairs
    Generates lists of files for train, val, and test
    """
    # Get file list
    with open(mapping_file, "r") as f:
        all_files = json.load(f)

    # Shuffle and split
    train_size = config.getfloat("dataset", "train_size")
    n_samples = len(all_files)
    # so that we can slice a dict consistently
    all_files = list(all_files.items())
    random.shuffle(all_files)
    train_files = dict(all_files[: int(train_size * n_samples)])
    val_test_files = all_files[int(train_size * n_samples) :]
    val_files = dict(val_test_files[: len(val_test_files) // 2])
    test_files = dict(val_test_files[len(val_test_files) // 2 :])

    # Dump
    train_path_out = Path(config["data_path"]["train_dir"]) / "train.json"
    _write_to_json(train_files, train_path_out)
    logger.info(f"Dumped {len(train_files)} training samples to {train_path_out}")

    val_path_out = Path(config["data_path"]["val_dir"]) / "val.json"
    _write_to_json(val_files, val_path_out)
    logger.info(f"Dumped {len(val_files)} val samples to {val_path_out}")

    test_path_out = Path(config["data_path"]["test_dir"]) / "test.json"
    _write_to_json(test_files, test_path_out)
    logger.info(f"Dumped {len(test_files)} test samples to {test_path_out}")


def make_vocab(file_mapping_path: str, config: t.Type[configparser.ConfigParser]):
    token2id_text = {
        "<pad>": 0,
        "<unk>": 1,
        "<long>": 2,
        "<mask>": 3,
        "<sos>": 4,
        "<eos>": 5,
    }
    token2id_action = copy.deepcopy(token2id_text)
    text_counter = len(token2id_text)
    action_counter = len(token2id_action)

    for i in range(127):
        character = chr(i).lower()
        if character not in token2id_text:
            token2id_text[character] = text_counter
            text_counter += 1

    file_mapping_path = Path(file_mapping_path)
    with open(file_mapping_path, "r") as f:
        file_mapping = json.load(f)
    for essay_paths in tqdm(list(file_mapping.values())[:100000]):
        with open(essay_paths[0], "r") as f:
            essay = json.load(f)
        keystroke_log = list(essay.values())[0]["logs"]
        for stroke in keystroke_log:
            text = stroke["dif"]["d"][0]["t"].lower()
            if len(text) > 1:
                text = "<long>"
            action = stroke["dif"]["d"][0]["e"]
            if text not in token2id_text:
                token2id_text[text] = text_counter
                text_counter += 1
            if action not in token2id_action:
                token2id_action[action] = action_counter
                action_counter += 1

    _write_to_json(token2id_text, config["resource_path"]["token2id_text_path"])
    _write_to_json(token2id_action, config["resource_path"]["token2id_action_path"])


def make_binary_classification_dataset(mapping_file: str, config: t.Type[configparser.ConfigParser]):
    """
    Read the json mapping for person - essay pairs
    Generate a binary classifcation dataset
    (each sample is a pair, label is whether it is a matching pair)
    """
    with open(Path(mapping_file), 'r') as f:
        all_files = json.load(f)
    all_files = list(all_files.values())
    correct_pairs = []
    incorrect_pairs = []
    idx = 0

    while len(correct_pairs) < 2501:
        essay_pair = all_files[idx]
        try:
            essay_1, essay_2 = essay_pair
            correct_pairs.append(
                {
                    'label': 1,
                    'essay_1': essay_1,
                    'essay_2': essay_2
                }
            )
        except BaseException as e:
            print(e)
            pass
        idx += 1

    idx += 1
    while len(incorrect_pairs) < 2501:
        essay_pair = all_files[idx]
        # essay_1, _ = essay_pair
        # essay_2 = all_files[idx + 300][1]
        # correct_pairs.append(
        #     {
        #         'label': 0,
        #         'essay_1': essay_1,
        #         'essay_2': essay_2
        #     }
        # )
        try:
            essay_1, _ = essay_pair
            essay_2 = all_files[idx + 20000][1]
            incorrect_pairs.append(
                {
                    'label': 0,
                    'essay_1': essay_1,
                    'essay_2': essay_2
                }
            )
        except BaseException as e:
            print(e)
            pass
        idx += 1

    correct_pairs.extend(incorrect_pairs)
    _write_to_jsonl(correct_pairs, "data/raw/all_binary_cls_data.jsonl")

    return correct_pairs


def main():
    print("hello world")
    config = configparser.ConfigParser()
    config.read("configs/config.ini")
    random.seed(config.getint("general", "seed"))

    # mapping = get_essay_pairs('/encrypted/keystrokes/INTERMEDIATE/GRE/', 'data/raw/all_data.jsonl')
    # train_val_test_split('data/raw/all_data.jsonl', config)
    # make_vocab("data/processed/train/train.json", config)
    make_binary_classification_dataset('data/raw/all_data.jsonl', config)


if __name__ == "__main__":
    main()
