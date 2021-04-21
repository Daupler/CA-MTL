from enum import Enum
from dataclasses import dataclass, field
from typing import List

from transformers import InputFeatures


class Split(Enum):
    train = "train"
    train_dev = "train_dev"
    dev = "dev"
    test = "test"


@dataclass(frozen=True)
class MultiTaskInputFeatures(InputFeatures):
    task_id: int = None


@dataclass
class MultiTaskDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        default=None,
        metadata={
            "help": "The input data dir. Should contain the task folders references in task_data_folders"
        }
    )
    tasks: List[str] = field(
        default=None,
        metadata={
            "help": "The task file that contains the tasks to train on. Must be provided"
        },
    )
    task_data_folders: List[str] = field(
        default=None,
        metadata={
            "help": "The task folders that contain the data for the tasks to train on. Should contain .tsv files for each split for the task. Must be provided"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

