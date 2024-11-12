import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import pdb

import numpy as np
import torch
import evaluate
import wandb
from datasets import load_metric
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
)
from dataHelper import get_dataset

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@dataclass
class ModelArguments:
    model_name_or_path: str 
    dropout: float
    exp: int


@dataclass
class DataArguments:
    dataset_name: str

def main():
    '''
    Initialize logging, seed, argparse...
    '''

    # Parse Arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set logging level
    logger.setLevel(logging.INFO)

    # Set seed
    set_seed(training_args.seed)

    # Initialize wandb
    if model_args.model_name_or_path == "allenai/scibert_scivocab_uncased":
        wandb.init(project=f"Task2-allenai_scibert_scivocab_uncased-{data_args.dataset_name}", 
                   config=training_args, 
                   name=f"allenai_scibert_scivocab_uncased-{data_args.dataset_name}_{model_args.exp}")
        training_args.output_dir = f"./results/allenai_scibert_scivocab_uncased-{data_args.dataset_name}"
    else:
        wandb.init(project=f"Task2-{model_args.model_name_or_path}-{data_args.dataset_name}", 
                   config=training_args, 
                   name=f"{model_args.model_name_or_path}-{data_args.dataset_name}_{model_args.exp}")

    # Log current arguments
    logger.info(f"Training/evaluation parameters {training_args}")

    '''
    Load datasets
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )
    datasets = get_dataset(data_args.dataset_name, tokenizer.sep_token)

    '''
    Load models
    '''
    num_labels = max(datasets['train']['label']) + 1
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        hidden_dropout_prob=model_args.dropout,
        attention_probs_dropout_prob=model_args.dropout,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    '''
    Trainer
    '''
    # Preprocess dataset
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False)
    tokenized_datasets = datasets.map(preprocess_function, batched=True)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Metrics
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = metric_accuracy.compute(predictions=predictions, references=labels)
        macro_f1 = metric_f1.compute(predictions=predictions, references=labels, average='macro')
        micro_f1 = metric_f1.compute(predictions=predictions, references=labels, average='micro')
        return {
            'accuracy': acc['accuracy'],
            'macro_f1': macro_f1['f1'],
            'micro_f1': micro_f1['f1'],
        }

    '''
    Initialize Trainer
    '''
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    trainer.train()
    trainer.evaluate()

    wandb.finish()

if __name__ == "__main__":
    main()