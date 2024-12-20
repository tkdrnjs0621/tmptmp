
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import re
from functools import partial
import logging
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import json
import accelerate
from accelerate import Accelerator

from accelerate import PartialState
from accelerate.utils import gather_object
query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

query_prompt_template = """
Question: {}

"""

messages_ner =  [{"role":"system","content":"You're a very effective entity extraction system."},
                         {"role":"user","content":query_prompt_one_shot_input},
                         {"role":"assistant","content":query_prompt_one_shot_output},
                         ]


messages_query_ner = [{"role": "system", "content": """You are a language model that perform entity recognition with the following text. Extract not just named entities, but all entities. Make sure they are in form of Name (Type). If it is not named entity, write in form of Name (noun). Also do not use initials and use full names."""},
#1
    {"role": "user", "content": """"Perform entity recognition for following text : Which film has the director born first, Two Weeks With Pay or Chhailla Babu?"""},
    {"role": "assistant", "content": """film (Noun)
director (Noun)    
Two Weeks With Pay (Film)
Chhallia Babu (Film)"""},
#2
    {"role": "user", "content":"""Perform entity recognition for following text : Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?"""},
    {"role": "assistant", "content": """Coolie No. 1 (Film)
film (Noun)
director (Noun)
nationality (Noun)
Sensational Trial (Film)"""},
#3
    {"role": "user", "content":"""Perform entity recognition for following text : What is the place of birth of Amparo Soler Leal's husband?"""},
    {"role": "assistant", "content": """Amparo Soler Leal (Person)
place of birth (Noun)
husband (Noun)"""}, 
#4
    {"role": "user", "content":"""Perform entity recognition for following text : Who was born first, Chris Campbell (Offensive Tackle) or Jacques Thieffry?"""},
    {"role": "assistant", "content": """Chris Campbell (Person)
Offensive Tackle (noun)
Jacques Thieffry (Person)"""}, 
#5
    {"role": "user", "content":"""Perform entity recognition for following text : Where did the composer of song Contigo En La Distancia die?"""},
    {"role": "assistant", "content": """Contigo En La Distancia (Song)
composer (noun)"""}, 
]


def build_prompt(example, tokenizer):
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return {"prompt": prompt, "prompt_tokens": prompt_tokens}


def decode(example, tokenizer, feature):
    text = tokenizer.decode(example[feature + "_ids"], skip_special_tokens=True)
    return {feature: text}

def map_messages_hipporag(row):
    txt = row["question"]
    messages = messages_ner+[{"role":"user","content":query_prompt_template.format(txt)}]
    row["messages"]=messages
    return row

def map_messages_entityrag(row):
    txt = row["question"]
    messages = messages_query_ner+[{"role":"user","content":"""Perform NER for following text : """+txt}]
    row["messages"]=messages
    return row

def map_entity(row):
    row['entities']=[t.strip() for t in row['output'].strip().split('\n')]
    return row

def set_file_handler(logger, path, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"):
    os.makedirs(os.path.dirname(path + "/run.log"), exist_ok=True)
    handler = logging.FileHandler(path + "/run.log")
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def extract_json_dict(text):
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}'
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return ''
    else:
        return ''

def map_json_dict(row):
    extracted = extract_json_dict(row["output"])
    row['entities']= extracted["named_entities"] if extracted!='' and 'named_entities' in extracted else ['']
    return row

def generate(model, tokenizer, dataloader,  **kwargs):
    output_ids = []
    cpp=[]
    with distributed_state.split_between_processes(dataloader) as mdataloader:
        for i, inputs in tqdm(enumerate(mdataloader, start=1), total=len(mdataloader)):
            inputs = inputs.to(distributed_state.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, pad_token_id=tokenizer.eos_token_id, **kwargs
                )
            cpp.extend(outputs[:, inputs["input_ids"].size(1):].tolist())
    # output_ids = gather_object(cpp)
    # return output_ids
    return cpp
def collate_fn(batch, tokenizer):
    prompts = [example["prompt"] for example in batch]
    inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    return inputs

# Updated main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument('--dataset_path', type=str, default='BeIR/msmarco')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Specific model name')
    parser.add_argument("--save_path", type=str, default="data/msmarco_er.jsonl", help="Path to save inference results")
    parser.add_argument("--ner_opt", choices=['entityrag', 'hipporag_original'], default='entityrag')
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processors for processing datasets")
    parser.add_argument("--max_tokens", type=int, default=300, help="Generation config; max new tokens")
    parser.add_argument("--do_sample", type=bool, default=False, help="Generation config; sampling flag")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation config; temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Generation config; top k")
    parser.add_argument("--top_p", type=float, default=0.1, help="Generation config; top p, nucleus sampling")
    args = parser.parse_args()

    # Initialize the accelerator
    
    distributed_state = PartialState()
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=distributed_state.device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Load dataset
    dataset = load_dataset(args.dataset_path, 'queries')['queries'].rename_column('text', 'question').select(range(1200))

    # Prepare dataset for NER options
    if args.ner_opt == 'hipporag_original':
        dataset = dataset.map(map_messages_hipporag)
    else:
        dataset = dataset.map(map_messages_entityrag)
    dataset = dataset.map(partial(build_prompt, tokenizer=tokenizer), num_proc=args.num_proc)

    # Prepare dataloader
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_proc,
        collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True
    )


    # Generate outputs
    output_ids = generate(
        model, tokenizer, dataloader,
        max_new_tokens=args.max_tokens, do_sample=args.do_sample,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
    )

    # Add and decode outputs
    dataset = dataset.add_column("output_ids", output_ids)
    dataset = dataset.map(partial(decode, tokenizer=tokenizer, feature="output"), num_proc=args.num_proc)

    # Post-process results
    if args.ner_opt == 'hipporag_original':
        dataset = dataset.map(map_json_dict, num_proc=args.num_proc)
    else:
        dataset = dataset.map(map_entity, num_proc=args.num_proc)

    # Select relevant columns and save results
    dataset = dataset.select_columns(["_id", "title", "question", "entities"])
    dataset.to_json(args.save_path, orient="records", lines=True)