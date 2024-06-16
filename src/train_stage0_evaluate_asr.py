
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import datasets
import evaluate
import torch

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, WhisperConfig, WhisperFeatureExtractor
from transformers.deepspeed import is_deepspeed_zero3_enabled
import numpy as np
import pandas as pd
from src.speech_text_paired_dataset import load_speech_text_paired_dataset, SpeechTextPairedDataCollator
from src.modeling_blsp import BlspModel
from src.modeling_whisper_encoder import WhisperEncoder
from datasets import load_metric
from transformers import GenerationConfig
logger = logging.getLogger(__name__)



generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=False,
    temperature=0.1,
    top_p=0.75,
    num_beams=1,
    num_return_sequences=1,
)

class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs,return_outputs=False):
        #print(inputs)
        speech_values=inputs.get("speech_values")
        speech_attention_mask=inputs.get("speech_attention_mask")
        input_ids=inputs.get("input_ids")
        suffix_input_ids=inputs.get("suffix_input_ids")
        output = model.generate(
                input_ids=input_ids,
                suffix_input_ids=suffix_input_ids,
                speech_values=speech_values,
                speech_attention_mask=speech_attention_mask,
                generation_config=generation_config,
            )
        return (None,output)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    llama_model: str = field(
        default=".model/model_v1", metadata={"help": "the path of base model"}
    )
    whisper_model: str = field(
        default=".model/model_v1", metadata={"help": "the path of whisper model"}
    )
    Blsp_model: str = field(default=".model/model_v1",metadata={"help":"vicuna+whisper+random_qformer"})
    safetensor: bool = field(default=False,metadata={"help":"vicuna+whisper+random_qformer"})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data: str = field(
        metadata={
            "help": "the root to load dataset"
        },
    )
    save_path: str = field(
        metadata={
            "help": "the root to load dataset"
        },
    ),
    manifest_files: str = field(
        default="",
        metadata={
            "help": "The name of the training unit text paired set split to use."
        },
    )
    instruction: str = field(
        default="",
        metadata={
            "help": "The text prefix instruction before speech input, default None"
        },
    )
    def evaluate(trainer):
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels



def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    #logger.info(f"Training/evaluation parameters {training_args}")
    #logger.info(f"Model parameters {model_args}")
    #logger.info(f"Dataset parameters {data_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_args.llama_model,padding_side='left')
    extractor = WhisperFeatureExtractor.from_pretrained(model_args.whisper_model)
    dataset = datasets.load_from_disk(data_args.data)
    model = BlspModel.from_pretrained(model_args.Blsp_model)
    generation_config.update(
        **{
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_pred = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(decoded_pred, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result={}
        with open(data_args.save_path, 'w', encoding='utf-8') as file:
            for pred, labels in zip(decoded_preds, decoded_labels):
                data = {"pred": pred, "labels": labels[0]}
                json.dump(data, file)
                file.write('\n')
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in decoded_pred]
        result["gen_len"] = np.mean(prediction_lens)

        total_lens=[np.count_nonzero(label != tokenizer.pad_token_id) for label in labels]
        result["total"] =np.mean(total_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result



    # 6. Define data collator
    data_collator = SpeechTextPairedDataCollator(
        pad_id=tokenizer.pad_token_id,
        sampling_rate=extractor.sampling_rate,
        extractor=extractor
    )


    # 7. Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 8. Evaluate
    evaluate(trainer)
    
    return 


if __name__ == "__main__":
    main()