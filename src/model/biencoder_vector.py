
import configparser

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import logging as t_logging


class KeystrokeBiencoder(PreTrainedModel):
    def __init__(self, autoconfig: AutoConfig, config: configparser.ConfigParser):
        super().__init__(autoconfig)
        if config:
            autoconfig.intermediate_size = config.getint('model', 'intermediate_size')
            autoconfig.num_hidden_layers = config.getint('model', 'n_layers')
            autoconfig.num_attention_heads = config.getint('model', 'n_attn_heads')
        self.key_proj = nn.Embedding(200, config.getint("model", "text_proj_size"))
        self.action_proj = nn.Embedding(25, config.getint("model", "action_proj_size"))
        # inverse or log
        self.timing_proj = nn.Linear(1, config.getint("model", "timing_proj_size"))
        # relative position
        self.position_proj = nn.Embedding(
            config.getint("model", "max_position"), config.getint("model", "position_proj_size")
        )
        self.pretrained_encoder = AutoModel.from_config(autoconfig)
        # for param in self.pretrained_encoder.embeddings.parameters():
        #     param.requires_grad = False

    def _score_candidates(self, query_logits, candidate_logits):
        return torch.matmul(query_logits, torch.transpose(candidate_logits, 0, 1))

    def _loss_fn(self, scores, labels):
        return F.cross_entropy(scores, labels)

    def forward(
        self,
        essay_key_sequences: torch.LongTensor,
        essay_action_sequences: torch.LongTensor,
        essay_timing_sequences: torch.FloatTensor,
        essay_position_sequences: torch.LongTensor,
        candidate_key_sequences: torch.LongTensor,
        candidate_action_sequences: torch.LongTensor,
        candidate_timing_sequences: torch.FloatTensor,
        candidate_position_sequences: torch.LongTensor,
        labels: torch.LongTensor,
        **kwargs
    ):
        essay_key_hidden = self.key_proj(essay_key_sequences)
        essay_action_hidden = self.action_proj(essay_action_sequences)
        essay_timing_hidden = self.timing_proj(essay_timing_sequences.unsqueeze(-1))
        essay_position_hidden = self.position_proj(essay_position_sequences)

        candidate_key_hidden = self.key_proj(candidate_key_sequences)
        candidate_action_hidden = self.action_proj(candidate_action_sequences)
        candidate_timing_hidden = self.timing_proj(candidate_timing_sequences.unsqueeze(-1))
        candidate_position_hidden = self.position_proj(candidate_position_sequences)

        essay_hidden = torch.cat(
            [
                essay_key_hidden,
                essay_action_hidden,
                essay_timing_hidden,
                essay_position_hidden,
            ],
            dim=2,
        )
        essay_attention_mask = (essay_hidden != 0).int()
        candidate_hidden = torch.cat(
            [
                candidate_key_hidden,
                candidate_action_hidden,
                candidate_timing_hidden,
                candidate_position_hidden,
            ],
            dim=2,
        )
        candidate_attention_mask = (candidate_hidden != 0).int()

        essay_logits = self.pretrained_encoder(
            inputs_embeds=essay_hidden,
            # attention_mask=essay_attention_mask
        ).last_hidden_state
        essay_logits = torch.mean(essay_logits, dim=1)
        essay_logits = F.normalize(essay_logits, dim=1)
        candidate_logits = self.pretrained_encoder(
            inputs_embeds=candidate_hidden,
            # attention_mask=candidate_attention_mask
        ).last_hidden_state
        candidate_logits = torch.mean(candidate_logits, dim=1)
        candidate_logits = F.normalize(candidate_logits, dim=1)

        scores = self._score_candidates(essay_logits, candidate_logits)
        loss = self._loss_fn(scores, labels) if torch.is_tensor(labels) else None

        return SequenceClassifierOutput(loss=loss, logits=scores)
