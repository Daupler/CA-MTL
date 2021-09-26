import os
import sys
import re
import json
import logging

import torch
from transformers import (
    HfArgumentParser,
    set_seed,
    EvalPrediction,
    BertConfig, 
    BertTokenizer
)

from src.model.ca_mtl import CaMtl, CaMtlArguments
from src.utils.misc import MultiTaskDataArguments, Split
from src.mtl_trainer import MultiTaskTrainer, MultiTaskTrainingArguments
from src.data.mtl_dataset import MultiTaskDataset
from src.data.task_dataset import TaskDataset

logger = logging.getLogger(__name__)


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )


def parse_cmd_args():
    parser = HfArgumentParser(
        (
            CaMtlArguments,
            MultiTaskDataArguments,
            MultiTaskTrainingArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()

    logger.info("Training/evaluation parameters %s", training_args)

    if model_args.encoder_type is None:
        model_args.encoder_type = model_args.model_name_or_path
        
    return model_args, data_args, training_args

def create_eval_datasets(mode, data_args, tokenizer, model_metadata):
    eval_datasets = {}
    for task_id, task_name in enumerate(model_metadata['tasks']):
        eval_datasets[task_name] = TaskDataset(
            task_name, task_id, data_args, tokenizer, mode=mode, 
            label_list=model_metadata['label_set'][task_name]
        )

    return eval_datasets


def main():
    model_args, data_args, training_args = parse_cmd_args()

    setup_logging(training_args)

    set_seed(training_args.seed)
    logger.info(training_args)

    model_metadata = json.load(open(f"{model_args.model_name_or_path}/metadata.json", 'r'))
    
    # update arguments with condition at model training
    data_args.max_seq_length = model_metadata['max_seq_length']
    num_tasks = len(model_metadata['label_set'])
    data_args.task_data_folders = data_args.task_data_folders*num_tasks
    data_args.tasks = model_metadata['tasks']
    model_args.encoder_type = model_metadata['model_name_or_path']
    
    config = BertConfig.from_pretrained(model_args.model_name_or_path)

    model = CaMtl.from_pretrained(
        model_args.model_name_or_path,
        model_args,
        data_args,
        config=config)

    logger.info(model)

    # load the tokenizer that was used when the model was trained
    tokenizer = BertTokenizer.from_pretrained(
        CaMtl.get_base_model(model_metadata['model_name_or_path']),
    )

    logger.info("Training tasks: %s", ", ".join([t for t in data_args.tasks]))

    trainer = MultiTaskTrainer(
        tokenizer,
        data_args,
        model=model,
        args=training_args,
        train_dataset=None,
        eval_datasets=None,
        test_datasets=create_eval_datasets(Split.test, data_args, tokenizer, model_metadata)
        if training_args.do_predict
        else None,
    )

    scoring_model = model_args.model_name_or_path.split("/")[-1]
    if training_args.do_predict:
        trainer.predict(scoring_model = scoring_model)
    


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
