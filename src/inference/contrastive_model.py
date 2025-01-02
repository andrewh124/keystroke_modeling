import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
import configparser
from tqdm import tqdm
import json
import typing as t
import random
import ast

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig
import gradio as gr
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _inference_setup(
    config: configparser.ConfigParser,
    checkpoint_dir: str
):
    from src.model.biencoder_vector import KeystrokeBiencoder
    from src.data.tokenizer_vector import KeystrokeTokenizer

    checkpoint_dir = Path(checkpoint_dir)

    logger.info(f"Loading model checkpoint from {checkpoint_dir}")
    pretrained_config = AutoConfig.from_pretrained(checkpoint_dir)

    # Init dataset
    tokenizer = KeystrokeTokenizer(config)

    # Init model
    model = KeystrokeBiencoder(pretrained_config, config)
    model.load_state_dict(
        torch.load(
            str(Path(f"{checkpoint_dir}/pytorch_model.bin")),
            map_location=torch.device('cpu')
        )
    )
    model.eval()

    return model, tokenizer


config = configparser.ConfigParser()
config.read('configs/config.ini')
model, tokenizer = _inference_setup(
    config,
    'training_out/contrastive_10_ep_relative_log_timing/checkpoint-15224'
)


def _generate_sample(n_candidates: str) -> t.Dict:
    n_candidates = int(n_candidates)
    data_path = Path('data/processed/val/val.json')
    with open(data_path, 'r') as f:
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

    random_samples = random.sample(data, n_candidates)
    out = {}
    out['input_essay'] = random_samples[0]['essay_1']
    out['gold_essay'] = random_samples[0]['essay_2']
    out['candidate_essays'] = []
    for sample in random_samples[1:]:
        out['candidate_essays'].append(sample['essay_1'])

    print('done')

    return out


def _process_to_batch(input_paths: t.Dict, tokenizer) -> t.Dict:
    def _process_keystroke_log(keystroke_stream: t.List, tokenizer) -> t.Dict:
        """
        Perform tokenization, conversion to ids, and truncation
        """
        key_sequence = []
        action_sequence = []
        timing_sequence = []
        t_0 = keystroke_stream[0]["t"] / 1000
        position_sequence = []

        for i_stroke, stroke in enumerate(keystroke_stream):
            # truncation
            if i_stroke >= 1024:
                break

            # convert to id
            key = stroke["dif"]["d"][0]["t"]
            action = stroke["dif"]["d"][0]["e"]
            time = stroke["t"]
            position = min(stroke["dif"]["d"][0]["p"], 2047)

            key_sequence.append(tokenizer.convert_token_to_id(key, "text"))
            action_sequence.append(tokenizer.convert_token_to_id(action, "action"))
            timing_diff = (time / 1000) - t_0
            t_0 = time / 1000
            timing_sequence.append(np.log(timing_diff + 0.0001))
            position_sequence.append(position)

        return {
            "key_sequence": torch.tensor(key_sequence[:1024], dtype=torch.long),
            "action_sequence": torch.tensor(action_sequence[:1024], dtype=torch.long),
            "timing_sequence": torch.tensor(timing_sequence[:1024], dtype=torch.float),
            "position_sequence": torch.tensor(position_sequence[:1024], dtype=torch.long),
        }

    gold_essay_path = input_paths['gold_essay']
    candidate_essay_paths = input_paths['candidate_essays']

    # process the input
    with open(input_paths['input_essay'], 'r', encoding='utf-8') as f:
        essay = json.load(f)
    input_essay = _process_keystroke_log(list(essay.values())[0]["logs"], tokenizer)

    # process the candidates
    candidate_essay_paths.insert(0, gold_essay_path)
    candidates = []
    for essay_path in candidate_essay_paths:
        with open(essay_path, 'r', encoding='utf-8') as f:
            candidate_essay = json.load(f)
        candidate_essay = _process_keystroke_log(list(candidate_essay.values())[0]["logs"], tokenizer)
        candidates.append(candidate_essay)

    essay_key_sequences = input_essay['key_sequence'].unsqueeze(0)
    essay_action_sequences = input_essay['action_sequence'].unsqueeze(0)
    essay_timing_sequences = input_essay['timing_sequence'].unsqueeze(0)
    essay_position_sequences = input_essay['position_sequence'].unsqueeze(0)

    candidate_key_sequences = []
    candidate_action_sequences = []
    candidate_timing_sequences = []
    candidate_position_sequences = []
    for candidate in candidates:
        candidate_key_sequences.append(candidate["key_sequence"])
        candidate_action_sequences.append(candidate["action_sequence"])
        candidate_timing_sequences.append(candidate["timing_sequence"])
        candidate_position_sequences.append(candidate["position_sequence"])
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
    }


def _predict_pipeline(input_paths):
    input_paths = ast.literal_eval(input_paths)
    batch = _process_to_batch(input_paths, tokenizer)
    with torch.no_grad():
        model_out = model(**batch)
    model_probs = torch.nn.functional.softmax(model_out.logits.squeeze()).cpu().numpy().tolist()

    probs = {f'essay {i_prob+1}': round(prob, 3) for i_prob, prob in enumerate(model_probs)}

    return probs


with gr.Blocks() as demo:
    with gr.Tab('Sample for logs'):
        n_candidates = gr.Slider(1, 16, step=1.0)
        sampled_paths = gr.Textbox()
        generate_sample = gr.Button('Generate sample')
    with gr.Tab('score essays'):
        paths = gr.Textbox()
        probs = gr.Label()
        score = gr.Button('Score similarity')

    generate_sample.click(_generate_sample, inputs=n_candidates, outputs=sampled_paths)
    score.click(_predict_pipeline, inputs=paths, outputs=probs)

demo.launch()



# @torch.no_grad()
# def inference(
#     model: PreTrainedModel, dataset: torch.utils.data.Dataset
# ) -> t.List[t.Dict]:
#     """
#     Run inference on dataset
#     Return a list of trait scores for each sample
#     """

#     dataloader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=False,
#         collate_fn=dataset.collate_fn,
#         pin_memory=True,
#     )
#     preds = []
#     out = []

#     for i_batch, batch in enumerate(tqdm(dataloader)):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         model_out = model(**batch)

#         preds.extend(model_out.logits.detach().cpu().numpy().tolist())

#     for pred in preds:
#         sample_inference = {
#             writing_trait: pred for writing_trait, pred in zip(SCORES, pred)
#         }
#         out.append(sample_inference)

#     return out
