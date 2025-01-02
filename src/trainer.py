import os

import torch
import torch.nn.functional as F
import numpy as np
from transformers import TrainingArguments, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW
from sklearn.metrics import accuracy_score, classification_report, roc_curve


def _equal_error_rate(labels, predictions):
    # https://stackoverflow.com/a/46026962
    fpr, tpr, _ = roc_curve(labels, predictions, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return eer


def compute_metrics(p):
    logits, labels = p
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    text_logits, text_labels = shift_logits[:, :, :110], shift_labels[:, :, 0]
    action_logits, action_labels = shift_logits[:, :, 110:135], shift_labels[:, :, 1]
    timing_logits, timing_labels = shift_logits[:, :, -2], shift_labels[:, :, 2]
    position_logits, position_labels = shift_logits[:, :, -1], shift_labels[:, :, 3]

    text_loss = F.cross_entropy(
        text_logits.reshape(-1, text_logits.size(-1)),
        text_labels.reshape(-1),
        label_smoothing=0.1,
    )
    action_loss = F.cross_entropy(
        action_logits.reshape(-1, action_logits.size(-1)),
        action_labels.reshape(-1),
        label_smoothing=0.1,
    )
    timing_loss = torch.log(F.mse_loss(timing_logits, timing_labels.float()))
    position_loss = torch.log(F.mse_loss(position_logits, position_labels.float()))

    return {
        "loss": (text_loss + action_loss + timing_loss + position_loss).item(),
        "text_loss": text_loss.item(),
        "action_loss": action_loss.item(),
        "timing_loss": timing_loss.item(),
        "position_loss": position_loss.item(),
    }


def compute_metrics_reduced(p):
    logits, labels = p
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:, :]

    text_logits = shift_logits[:, :, :110].contiguous()
    text_labels = shift_labels[:, :, 0].contiguous()
    action_logits = shift_logits[:, :, 110:135].contiguous()
    action_labels = shift_labels[:, :, 1].contiguous()

    text_loss = F.cross_entropy(
        text_logits.view(-1, text_logits.size(-1)), text_labels.view(-1)
    )
    action_loss = F.cross_entropy(
        action_logits.view(-1, action_logits.size(-1)), action_labels.view(-1)
    )

    return {
        "loss": (text_loss + action_loss).item(),
        "text_loss": text_loss.item(),
        "action_loss": action_loss.item(),
    }


def compute_metrics_reduced_v2(p):
    logits, labels = p
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:, :]

    text_logits = shift_logits[:, :, :110].contiguous()
    text_labels = shift_labels[:, :, 0].contiguous()
    action_logits = shift_logits[:, :, 110:135].contiguous()
    action_labels = shift_labels[:, :, 1].contiguous()
    aux_logits = shift_logits[:, :, -1].contiguous()
    aux_labels = shift_labels[:, :, 2].contiguous()

    text_loss = F.cross_entropy(
        text_logits.view(-1, text_logits.size(-1)), text_labels.view(-1)
    )
    action_loss = F.cross_entropy(
        action_logits.view(-1, action_logits.size(-1)), action_labels.view(-1)
    )
    aux_loss = torch.log(F.mse_loss(aux_logits, aux_labels))

    return {
        "loss": text_loss + action_loss + aux_loss,
        "text_loss": text_loss,
        "action_loss": action_loss,
        "aux_loss": aux_loss,
    }


def compute_metrics_contrastive(p):
    logits, labels = p

    np.save('job-logs/logits.npy', logits)
    np.save('job-logs/labels.npy', labels)

    predictions = np.argmax(logits, axis=-1)
    logits = torch.from_numpy(logits)
    probabilities = F.softmax(logits.float(), dim=1).numpy()
    probabilities = probabilities[:, 1]
    accuracy = accuracy_score(labels, predictions)
    eer = _equal_error_rate(labels, probabilities)

    return {
        'accuracy': accuracy,
        'eer': eer
    }


def init_transformers_training_args(config, model):
    config = config["training"]

    if config['optim'] == 'adafactor':
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None
        )
        scheduler = AdafactorSchedule(optimizer)
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config.getfloat("learning_rate")
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.getint("warmup_steps"),
            num_training_steps=config.getint('num_training_steps')
        )

    training_args = TrainingArguments(
        run_name=config["run_name"],
        output_dir=os.path.join(config["output_dir"], config["run_name"]),
        per_device_train_batch_size=config.getint("train_batch_size"),
        per_device_eval_batch_size=config.getint("eval_batch_size"),
        # optim=config["optim"],
        # lr_scheduler_type=config["lr_scheduler_type"],
        # learning_rate=config.getfloat("learning_rate"),
        # warmup_steps=config.getint("warmup_steps"),
        num_train_epochs=config.getint("n_epochs"),
        save_total_limit=config.getint("n_ckpt_limit"),
        seed=config.getint("seed"),
        gradient_accumulation_steps=config.getint("gradient_accumulation_steps"),
        dataloader_num_workers=config.getint("n_workers"),
        fp16=config.getboolean("fp16"),
        no_cuda=config.getboolean("no_cuda"),
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        log_level="info",
        save_strategy="epoch",
        gradient_checkpointing=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        disable_tqdm=False,
        report_to=["tensorboard"],
        dataloader_drop_last=True,
        ignore_data_skip=True
    )

    return training_args, optimizer, scheduler
