import argparse
import configparser
import os
from pathlib import Path

import torch
from transformers import Trainer
from transformers import AutoConfig

from src.data.tokenizer_vector import KeystrokeTokenizer
from src.data.dataset_biencoder_vector import KeystrokeDataset
from src.model.biencoder_vector import KeystrokeBiencoder
from src.trainer import (
    init_transformers_training_args,
    compute_metrics_contrastive
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def training_setup(config):
    tokenizer = KeystrokeTokenizer(config)
    train_dataset = KeystrokeDataset(config, tokenizer, "train")
    val_dataset = KeystrokeDataset(config, tokenizer, "valid")
    autoconfig = AutoConfig.from_pretrained(config['model']['pretrained_name'])
    model = KeystrokeBiencoder(autoconfig, config)
    model.load_state_dict(torch.load(str(Path("training_out/contrastive_10_ep_relative_log_timing/checkpoint-15224/pytorch_model.bin"))))
    training_args, optimizer, scheduler = init_transformers_training_args(config, model)

    return model, optimizer, scheduler, train_dataset, val_dataset, training_args


if __name__ == "__main__":
    # qsub -l h=lonf train.sh
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--run_name", type=str, required=False)
    argparser.add_argument("--n_epochs", type=int, required=False)
    argparser.add_argument("--train_batch_size", type=int, required=False)
    argparser.add_argument("--eval_batch_size", type=int, required=False)
    argparser.add_argument("--fp16", type=bool, required=False)
    argparser.add_argument("--no_cuda", type=bool, required=False)
    args = vars(argparser.parse_args())

    config = configparser.ConfigParser()
    config.read(os.path.join("configs", "config.ini"))
    for arg, value in args.items():
        if value:
            config.set("training", arg, str(value))

    (
        model,
        optimizer,
        scheduler,
        train_dataset,
        val_dataset,
        training_args,
    ) = training_setup(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=train_dataset.collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_contrastive,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(metrics)
