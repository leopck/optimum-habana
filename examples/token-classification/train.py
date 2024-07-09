#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import evaluate
from functools import partial
from datasets import load_dataset, Features, Sequence, ClassLabel, Value, Array2D
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
)
import json
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as hf_logging
from transformers.utils.versions import require_version
from transformers import LiltForTokenClassification


# Check if HPU is available
try:
    from optimum.habana import GaudiTrainer, GaudiTrainingArguments
    HPU_AVAILABLE = True
except ImportError:
    HPU_AVAILABLE = False

# Check if CUDA is available
import torch
CUDA_AVAILABLE = torch.cuda.is_available()

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"})


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."})
    max_predict_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of prediction examples to this value if set."})


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GaudiTrainingArguments if HPU_AVAILABLE else TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    hf_logging.set_verbosity_info()
    logger.setLevel(training_args.get_process_log_level())
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    logger.info(f"Training Args: {training_args}")
    logger.info(f"Data Args: {data_args}")
    logger.info(f"Model Args: {model_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    dataset = load_dataset(data_args.dataset_name)

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    logger.info(f"Test dataset size: {len(dataset['test'])}")

    labels = dataset['train'].features['ner_tags'].feature.names
    logger.info(f"Available labels: {labels}")

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    model_id = model_args.model_name_or_path
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)

    features = Features({
        "input_ids": Sequence(feature=Value(dtype="int64")),
        "attention_mask": Sequence(feature=Value(dtype="int64")),
        "bbox": Array2D(dtype="int64", shape=(512, 4)),
        "labels": Sequence(ClassLabel(names=labels)),
    })

    def process(sample, processor=None):
        encoding = processor(
            sample["image"].convert("RGB"),
            sample["tokens"],
            boxes=sample["bboxes"],
            word_labels=sample["ner_tags"],
            padding="max_length",
            truncation=True,
        )
        del encoding["pixel_values"]
        return encoding

    proc_dataset = dataset.map(
        partial(process, processor=processor),
        remove_columns=["image", "tokens", "ner_tags", "id", "bboxes"],
        features=features,
    ).with_format("torch")

    logger.info(proc_dataset["train"].features.keys())

    model = LiltForTokenClassification.from_pretrained(
        model_id, num_labels=len(labels), label2id=label2id, id2label=id2label
    )

    metric = evaluate.load("seqeval")
    ner_labels = list(model.config.id2label.values())

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        all_predictions = []
        all_labels = []
        for prediction, label in zip(predictions, labels):
            for predicted_idx, label_idx in zip(prediction, label):
                if label_idx == -100:
                    continue
                all_predictions.append(ner_labels[predicted_idx])
                all_labels.append(ner_labels[label_idx])
        return metric.compute(predictions=[all_predictions], references=[all_labels])

    if HPU_AVAILABLE:
        trainer = GaudiTrainer(
            model=model,
            args=training_args,
            train_dataset=proc_dataset["train"],
            eval_dataset=proc_dataset["test"],
            compute_metrics=compute_metrics,
        )
    else:
        device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        model.to(device)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=proc_dataset["train"],
            eval_dataset=proc_dataset["test"],
            compute_metrics=compute_metrics,
        )

    if training_args.do_train:
        logger.info("*** Training ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(proc_dataset["train"])
        metrics["train_samples"] = min(max_train_samples, len(proc_dataset["train"]))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(proc_dataset["test"])
        metrics["eval_samples"] = min(max_eval_samples, len(proc_dataset["test"]))
        for key, value in metrics.items():
            if isinstance(value, dict):
                logger.info(f"{key}: {json.dumps(value, indent=2)}")
            else:
                logger.info(f"{key}: {value}")
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
