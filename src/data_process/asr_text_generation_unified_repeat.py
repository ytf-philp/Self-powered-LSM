import logging
import os
import sys
import json
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union, Tuple
import argparse
import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass
from transformers import LlamaTokenizer, LlamaForCausalLM
import random


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("continue writing")


#Text_Format = (
#    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \n\n"
#    "USER: {instruct} (no more than 100 words): {input} \n"
#    "ASSISTANT:"
#)
Text_Format = (
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \n\n"
    "USER: {instruct}: {input} \n"
    "ASSISTANT:"
)


def get_shard_range(tot, nshard, rank,start):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_dataset(manifest, nshard, rank,start=None):
    with open(manifest, "r") as f:
        lines = f.readlines()
        start, end = get_shard_range(len(lines), nshard, rank,start)
        lines = lines[start:end]
        lines = [json.loads(line.strip()) for line in lines]
    dataset = Dataset.from_list(lines)


    return dataset


def collate_tokens(
        values: List[List[int]],
        pad_id: int
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(torch.LongTensor(v), res[i][-len(v):])

    return res


@dataclass
class DataCollator:
    pad_id: int = 0

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        audio = [sample["audio"] for sample in samples]
        instruction = [sample["instruction"] for sample in samples]
        

        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio": audio,
            "instruction":instruction
        }
    

def continue_writing(
    llm_path,
    instructions,
    manifest,
    lab_dir,
    nshard=24,
    rank=0,
    batch_size=16,
    cuda = "cuda:1",
    start = None
):
    accelerator = Accelerator()
    logger.info(accelerator.state)
    device = accelerator.device
    #datasets.load_from_disk("/self-powered/data/100-disk")
    dataset = get_dataset(manifest, nshard, rank,start)
    #with open(manifest, 'r', encoding='utf-8') as f:
    #    transcriptions = json.load(f)
    #dataset = Dataset.from_list(transcriptions)
    tokenizer = LlamaTokenizer.from_pretrained(llm_path)

    def process_dataset(batch):
        instruction = random.choice(instructions)
        batch["input_ids"] = tokenizer(Text_Format.format(instruct=instruction,input=batch["text"])).input_ids
        batch["attention_mask"] = [1] * len(batch["input_ids"])
        batch["audio"] = batch["audio"]
        batch["instruction"] = instruction
        batch["length"] = len(batch["input_ids"])
        return batch
    
    def is_in_length_range(length):
            return length > 0 and length < 150
    
    dataset = dataset.map(process_dataset)
    dataset = dataset.filter(is_in_length_range, input_columns=["length"])

    model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16)

    data_collator = DataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )

    ### prepare everything
    model, dataloader = accelerator.prepare(
        model, dataloader
    )
    model.to(device)
    model.eval()

    split = os.path.splitext(os.path.basename(manifest))[0]
    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.jsonl"

    os.makedirs(lab_dir, exist_ok=True)

    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    with open(lab_path, "w", encoding='utf-8') as f:
        for batch in dataloader:
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=225,
                do_sample=False,
                num_beams=1,
                top_p=0.75,
                temperature=0.1,
                num_return_sequences=1,
            )
            input_length = batch["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for audio, text,instruction in zip(batch["audio"], output_text,batch["instruction"]):
                json_string = json.dumps(
                    {
                        "instruction":instruction,
                        "audio": audio,
                        "text": text
                    },ensure_ascii=False
                )
                print(json_string, file=f)
            progress_bar.update(1)

    logger.info("finished successfully")
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--manifest', type=str)
    parser.add_argument('--nshard', type=int,default=1)
    parser.add_argument('--cate', type=str,default="translation")
    parser.add_argument('--rank', type=int,default=0)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()
    
    
    instructions = ["Repeat the provided text, ensuring to maintain its original meaning and details",
                    "Rephrase the text without altering its initial intent and key information", 
                    "Paraphrase the provided text while preserving all original facts and nuances", 
                    "Echo the content of the text, maintaining its exact purpose and details", 
                    "Retell the given information without changing its meaning or losing any critical data"]
    
    continue_writing(llm_path="/self-powered/model/vicuna-7b-v1.5",
                     instructions=instructions,
                     manifest=args.manifest,
                     lab_dir="/self-powered/data/new_aug/data_aug_"+args.cate,
                     nshard=args.nshard,rank=args.rank,batch_size=32,
                     start=None)




