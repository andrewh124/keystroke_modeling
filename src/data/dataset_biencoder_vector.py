import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import configparser
import glob
import json
import logging
import typing as t
from datetime import datetime

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KeystrokeDataset(Dataset):
    """
    Torch dataset with lazy loading for contrastive training
    """

    def __init__(
        self,
        config: t.Type[configparser.ConfigParser],
        tokenizer: t.Type[AutoTokenizer],
        mode: str = "train",
    ):
        self.config = config
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = self.config.getint("dataset", "seq_len")

        self.data = self._read_data()
        self.n_samples = len(self.data)

        logger.info(f" Loaded {self.n_samples} {self.mode} samples")

    def _read_data(self):
        if "train" in self.mode:
            input_path = self.config["data_path"]["train"]
        elif self.mode == "valid":
            input_path = self.config["data_path"]["val"]
        elif self.mode == "test":
            input_path = self.config["data_path"]["test"]
        input_path = Path(input_path)

        with open(input_path, "r") as f:
            file_mapping = json.load(f)

        data = []
        for user, essay_pair_paths in file_mapping.items():
            try:
                sample = {
                    "user_id": user,
                    "essay_1": essay_pair_paths[0],
                    "essay_2": essay_pair_paths[1],
                }
                data.append(sample)
            except IndexError:
                pass

        if self.mode == "train_subset":
            data = data[:200]

        return data

    def _convert_to_sample(self, sample: t.Dict) -> t.Dict:
        """
        Load the files from the paths from file mapping dataset
        Each sample includes 1 input essay and 1 gold candidate with optional hard candidates

        Note: this is not picking which essay type to be gold
        """
        try:
            essay_1_path = Path(sample["essay_1"])
            with open(essay_1_path, "r", encoding="utf-8") as f:
                essay_1 = json.load(f)
            essay_1 = self._process_keystroke_log(list(essay_1.values())[0]["logs"])

            essay_2_path = Path(sample["essay_2"])
            with open(essay_2_path, "r", encoding="utf-8") as f:
                essay_2 = json.load(f)
            essay_2 = self._process_keystroke_log(list(essay_2.values())[0]["logs"])
        except BaseException:
            # essay sample is corrupted
            return None

        return {'user_id': sample['user_id'], "input_essay": essay_1, "candidate_essays": [essay_2]}

    def _process_keystroke_log(self, keystroke_stream: t.List) -> t.Dict:
        """
        Perform tokenization, conversion to ids, and truncation
        Relative timing, absolute position
        """
        key_sequence = []
        action_sequence = []
        timing_sequence = []
        position_sequence = []
        t_0 = keystroke_stream[0]["t"] / 1000

        for i_stroke, stroke in enumerate(keystroke_stream):
            # truncation
            if i_stroke >= self.max_len:
                break

            # convert to id
            key = stroke["dif"]["d"][0]["t"]
            action = stroke["dif"]["d"][0]["e"]
            time = stroke["t"]
            position = stroke["dif"]["d"][0]["p"]

            if (
                (not isinstance(key, str))
                or (not isinstance(action, str))
                or (not isinstance(time, int))
                or (not isinstance(position, int))
            ):
                continue

            key_sequence.append(self.tokenizer.convert_token_to_id(key, "text"))
            action_sequence.append(self.tokenizer.convert_token_to_id(action, "action"))
            timing_diff = max((time / 1000) - t_0, 0)
            t_0 = time / 1000
            timing_sequence.append(np.log(timing_diff + 0.0001))
            position = min(position, self.config.getint("model", "max_position") - 1)
            position_sequence.append(position)

        return {
            "key_sequence": torch.tensor(key_sequence[:self.max_len], dtype=torch.long),
            "action_sequence": torch.tensor(action_sequence[:self.max_len], dtype=torch.long),
            "timing_sequence": torch.tensor(timing_sequence[:self.max_len], dtype=torch.float),
            "position_sequence": torch.tensor(position_sequence[:self.max_len], dtype=torch.long),
        }

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = self.data[index]
        training_sample = self._convert_to_sample(sample)
        # when essay sample is corrupted
        while training_sample is None:
            index = np.random.randint(0, self.n_samples - 1)
            training_sample = self._convert_to_sample(self.data[index])

        return training_sample

    def collate_fn(self, batch: t.Tuple[t.Dict]):
        essay_key_sequences = []
        essay_action_sequences = []
        essay_timing_sequences = []
        essay_position_sequences = []

        candidate_key_sequences = []
        candidate_action_sequences = []
        candidate_timing_sequences = []
        candidate_position_sequences = []

        user_ids = []

        for sample in batch:
            essay_key_sequences.append(sample["input_essay"]["key_sequence"])
            essay_action_sequences.append(sample["input_essay"]["action_sequence"])
            essay_timing_sequences.append(sample["input_essay"]["timing_sequence"])
            essay_position_sequences.append(sample["input_essay"]["position_sequence"])

            for candidate in sample["candidate_essays"]:
                candidate_key_sequences.append(candidate["key_sequence"])
                candidate_action_sequences.append(candidate["action_sequence"])
                candidate_timing_sequences.append(candidate["timing_sequence"])
                candidate_position_sequences.append(candidate["position_sequence"])

            user_ids.append(sample['user_id'])

        essay_key_sequences = pad_sequence(essay_key_sequences, batch_first=True)
        essay_action_sequences = pad_sequence(essay_action_sequences, batch_first=True)
        essay_timing_sequences = pad_sequence(essay_timing_sequences, batch_first=True)
        essay_position_sequences = pad_sequence(
            essay_position_sequences, batch_first=True
        )

        candidate_key_sequences = pad_sequence(
            candidate_key_sequences, batch_first=True
        )
        candidate_action_sequences = pad_sequence(
            candidate_action_sequences, batch_first=True
        )
        candidate_timing_sequences = pad_sequence(
            candidate_timing_sequences, batch_first=True
        )
        candidate_position_sequences = pad_sequence(
            candidate_position_sequences, batch_first=True
        )

        # Make label vec for in-batch negatives
        essay_bs = essay_key_sequences.size(0)
        candidate_bs = candidate_key_sequences.size(0)
        candidate_group = int(candidate_bs / essay_bs)
        labels = torch.arange(0, candidate_bs, candidate_group)

        # Note: handle attention_mask inside the model forward()
        # so that we can use this for multiple model input types
        # as the mask can have mismatch shape
        return {
            "essay_key_sequences": essay_key_sequences,
            "essay_action_sequences": essay_action_sequences,
            "essay_timing_sequences": essay_timing_sequences,
            "essay_position_sequences": essay_position_sequences,
            "candidate_key_sequences": candidate_key_sequences,
            "candidate_action_sequences": candidate_action_sequences,
            "candidate_timing_sequences": candidate_timing_sequences,
            "candidate_position_sequences": candidate_position_sequences,
            "labels": labels,
            'user_ids': user_ids
        }
