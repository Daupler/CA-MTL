<h1 align="center">
<p>Conditional Adaptive Multi-Task Learning</p>
</h1>

<p align="center">
    <a>
        <img alt="Python" src="https://img.shields.io/badge/Python-3.7-blue">
    </a>
    <a>
        <img alt="Python" src="https://img.shields.io/badge/Pytorch-1.6-blue">
    </a>
    <a>
        <img alt="Python" src="https://img.shields.io/badge/Release-1.0.0-blue">
    </a>
       <a>
        <img alt="Python" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>


The source code uses the [huggingface implementation](https://github.com/huggingface/transformers) of transformers adapted for multitask training.


## Requirements

- Python 3.7
- Pytorch 1.6
- Huggingface transformers 2.11.0 

Note: Newer versions of the requirements should work, but was not tested.

### Using a virual environment 

```bash
# Create a virtual environment
python3.7 -m venv ca_mtl_env
source ca_mtl_env/bin/activate 

# Update pip to meet sentencepiece requirement
pip install -U pip

# Install the requirements
pip install requirements-with-torch.txt
# If you are using an environment that have torch already installed use "requirements.txt"
```


### Using Docker

We created a public docker image that contains all the requirements and the source code hosted on DockerHub.

```bash
export DOCKER_IMG=DOCKER_FILE:latest

# Pull the image
docker pull $DOCKER_IMG
```

## Data

Using the [official GLUE data download scirpt](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py), download all datasets 

```bash
export DATA_DIR=/my/data/dir/path
# Make sure the data folder exists

python download_glue_data.py --data_dir $DATA_DIR --tasks all
```

## Run training 

```bash
# Set the output dir
export OUTPUT_DIR=/path/to/output/dir
```

### Using the created virtual environment
```bash
python run.py --model_name_of_path CA-MTL-base --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --do_train
```

### Using the pulled docker image
```bash
docker run -v /data:$DATA_DIR $DOCKER_IMG --model_name_of_path CA-MTL-base --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --do_train

```

### Usage 
```
usage: run.py [-h] --model_name_or_path MODEL_NAME_OR_PATH --data_dir DATA_DIR
              [--encoder_type]
              [--tasks TASKS [TASKS ...]] [--task_data_folders TASKS [TASK_DATA_FOLDERS ...]] 
              [--overwrite_cache]
              [--max_seq_length MAX_SEQ_LENGTH] --output_dir OUTPUT_DIR
              [--overwrite_output_dir] [--do_train] [--do_eval] [--do_predict]
              [--evaluate_during_training]
              [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
              [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
              [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
              [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
              [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
              [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
              [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM]
              [--num_train_epochs NUM_TRAIN_EPOCHS] [--max_steps MAX_STEPS]
              [--warmup_steps WARMUP_STEPS] [--logging_dir LOGGING_DIR]
              [--logging_first_step] [--logging_steps LOGGING_STEPS]
              [--save_steps SAVE_STEPS] [--save_total_limit SAVE_TOTAL_LIMIT]
              [--no_cuda] [--seed SEED] [--fp16]
              [--fp16_opt_level FP16_OPT_LEVEL] [--local_rank LOCAL_RANK]
              [--tpu_num_cores TPU_NUM_CORES] [--tpu_metrics_debug]
              [--use_mt_uncertainty]

arguments:
  -h, --help            show this help message and exit
 --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from: CA-
                        MTL-base, CA-MTL-large, bert-base-cased bert-base-
                        uncased, bert-large-cased, bert-large-uncased
  --data_dir DATA_DIR   The input data dir. Should contain the .tsv files (or
                        other data files) for the task.
  --encoder-name        The string corresponding to the encoder type to be utilized by CAMtl
                        This will default to the model_name_or_path if not passes. Only required
                        when scoring a previously fine-tuned model
  --tasks TASKS [TASKS ...]
                        The task file that contains the tasks to train on. If
                        None all tasks will be used
  --task_data_folders TASK_DATA_FOLDERS [TASK_DATA_FOLDERS ...]
                        The folders where the data lies for your tasks. This will be the folder
                        under your DATA_DIR
  --overwrite_cache     Overwrite the cached training and evaluation sets
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        tokenization. Sequences longer than this will be
                        truncated, sequences shorter will be padded.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --overwrite_output_dir
                        Overwrite the content of the output directory.Use this
                        to continue training if output_dir points to a
                        checkpoint directory.
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the dev set.
  --do_predict          Whether to run predictions on the test set.
  --evaluate_during_training
                        Run evaluation during training at each logging step.
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for training.
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for evaluation.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Deprecated, the use of `--per_device_train_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        training.
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Deprecated, the use of `--per_device_eval_batch_size`
                        is preferred.Batch size per GPU/TPU core/CPU for
                        evaluation.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --logging_dir LOGGING_DIR
                        Tensorboard log dir.
  --logging_first_step  Log and eval the first global_step
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --save_total_limit SAVE_TOTAL_LIMIT
                        Limit the total amount of checkpoints.Deletes the
                        older checkpoints in the output_dir. Default is
                        unlimited checkpoints
  --no_cuda             Do not use CUDA even when it is available
  --seed SEED           random seed for initialization
  --fp16                Whether to use 16-bit (mixed) precision (through
                        NVIDIA apex) instead of 32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --tpu_num_cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by
                        launcher script)
  --tpu_metrics_debug   TPU: Whether to print debug metrics
  --use_mt_uncertainty  Use MT-Uncertainty sampling method
```

Since our code is based on the [huggingface implementation](https://github.com/huggingface/transformers). 
All parameters are described in their [documentation](https://huggingface.co/transformers/main_classes/trainer.html?highlight=trainer)

## License

MIT

## How do I cite CA-MTL ?
```
@inproceedings{
    pilault2021conditionally,
    title={Conditionally Adaptive Multi-Task Learning: Improving Transfer Learning in {\{}NLP{\}} Using Fewer Parameters {\&} Less Data},
    author={Jonathan Pilault and Amine El hattami and Christopher Pal},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=de11dbHzAMF}
}
```

## Contact and Contribution
For any question or request, please create a Github issue in this repository.



