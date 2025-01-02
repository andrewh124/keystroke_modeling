import json
import os
import configparser


class KeystrokeTokenizer:
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.max_len = int(self.config["dataset"]["seq_len"])
        self.token2id_text, self.token2id_action = self._init_vocab(self.config)
        self.id2token_text = {v: k for k, v in self.token2id_text.items()}
        self.id2token_action = {v: k for k, v in self.token2id_action.items()}

        self.mask_token_id = self.token2id_text["<mask>"]
        self.pad_token_id = self.token2id_text["<pad>"]
        self.long_sequence_id = self.token2id_text["<long>"]
        self.unk_token_id = self.token2id_text["<unk>"]
        self.all_special_ids = [
            self.mask_token_id,
            self.pad_token_id,
            self.long_sequence_id,
            self.unk_token_id,
        ]

    def _init_vocab(self, config):
        text_vocab_path = config["resource_path"]["token2id_text_path"]
        with open(text_vocab_path, "r") as f:
            token2id_text = json.load(f)

        action_vocab_path = config["resource_path"]["token2id_action_path"]
        with open(action_vocab_path, "r") as f:
            token2id_action = json.load(f)

        return token2id_text, token2id_action

    def encode(self, input_seq, token_type, truncation=False):
        if token_type == "text":
            token2id = self.token2id_text
        elif token_type == "action":
            token2id = self.token2id_action

        if truncation:
            input_seq = input_seq[: self.max_len]
        encoded_seq = [token2id.get(tok, 1) for tok in input_seq]

        return encoded_seq

    def batch_encode(self, input_seq_batch, truncation=False):
        return [self.encode(seq, truncation) for seq in input_seq_batch]

    def decode(self, input_seq, token_type):
        if token_type == "text":
            id2token = self.id2token_text
        elif token_type == "action":
            id2token = self.id2token_action

        return [id2token.get(id_, "<unk>") for id_ in input_seq]

    def convert_id_to_token(self, input_id, token_type):
        if token_type == "text":
            id2token = self.id2token_text
        elif token_type == "action":
            id2token = self.id2token_action

        return id2token.get(input_id, "<unk>")

    def convert_token_to_id(self, input_token, token_type):
        token2id = self.token2id_text if token_type == "text" else self.token2id_action

        return token2id.get(input_token, 1)

    def create_special_token_mask(self, input_ids):
        # monkey see monkey do
        # https://github.com/huggingface/transformers/blob/b487096b02307cd6e0f132b676cdcc7255fe8e74/src/transformers/tokenization_utils_base.py#L3408
        all_special_ids = self.all_special_ids
        special_tokens_mask = [
            1 if token_id in all_special_ids else 0 for token_id in input_ids
        ]

        return special_tokens_mask


if __name__ == "__main__":
    import configparser

    config = configparser.ConfigParser()
    config.read(os.path.join("configs", "config.ini"))
    tokenizer = KeystrokeTokenizer(config)
    print(tokenizer.token2id)
