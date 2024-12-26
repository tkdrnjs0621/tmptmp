import argparse
import json
import logging
import os
import re
from functools import partial

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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

messages_ner = [
    {"role": "system", "content": "You're a very effective entity extraction system."},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
]


messages_query_ner = [
    {"role": "system", "content": """You are a language model that perform entity recognition with the following text. Extract not just named entities, but all entities. Make sure they are in form of Name (Type). If it is not named entity, write in form of Name (noun). Also do not use initials and use full names."""},
    # 1
    {"role": "user", "content": """"Perform entity recognition for following text : Which film has the director born first, Two Weeks With Pay or Chhailla Babu?"""},
    {
        "role": "assistant",
        "content": """film (Noun)
director (Noun)
Two Weeks With Pay (Film)
Chhallia Babu (Film)""",
    },
    # 2
    {"role": "user", "content": """Perform entity recognition for following text : Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?"""},
    {
        "role": "assistant",
        "content": """Coolie No. 1 (Film)
film (Noun)
director (Noun)
nationality (Noun)
Sensational Trial (Film)""",
    },
    # 3
    {"role": "user", "content": """Perform entity recognition for following text : What is the place of birth of Amparo Soler Leal's husband?"""},
    {
        "role": "assistant",
        "content": """Amparo Soler Leal (Person)
place of birth (Noun)
husband (Noun)""",
    },
    # 4
    {"role": "user", "content": """Perform entity recognition for following text : Who was born first, Chris Campbell (Offensive Tackle) or Jacques Thieffry?"""},
    {
        "role": "assistant",
        "content": """Chris Campbell (Person)
Offensive Tackle (noun)
Jacques Thieffry (Person)""",
    },
    # 5
    {"role": "user", "content": """Perform entity recognition for following text : Where did the composer of song Contigo En La Distancia die?"""},
    {
        "role": "assistant",
        "content": """Contigo En La Distancia (Song)
composer (noun)""",
    },
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
    messages = messages_ner + [{"role": "user", "content": query_prompt_template.format(txt)}]
    row["messages"] = messages
    return row


def map_messages_entityrag(row):
    txt = row["question"]
    messages = messages_query_ner + [{"role": "user", "content": """Perform NER for following text : """ + txt}]
    row["messages"] = messages
    return row


def map_entity(row):
    row["entities"] = [t.strip() for t in row["output"].strip().split("\n")]
    return row


def set_file_handler(logger, path, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"):
    os.makedirs(os.path.dirname(path + "/run.log"), exist_ok=True)
    handler = logging.FileHandler(path + "/run.log")
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def extract_json_dict(text):
    pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}"
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return ""
    else:
        return ""


def map_json_dict(row):
    extracted = extract_json_dict(row["output"])
    row["entities"] = extracted["named_entities"] if extracted != "" and "named_entities" in extracted else [""]
    return row


def generate(model, tokenizer, dataloader, accelerator, generation_config):
    output_ids = []
    for i, inputs in tqdm(enumerate(dataloader, start=1), total=len(dataloader)):
        inputs = accelerator.prepare(inputs)  # Ensure inputs are moved to the correct device
        with torch.no_grad():
            outputs = accelerator.unwrap_model(model).generate(**inputs, pad_token_id=tokenizer.eos_token_id, **generation_config)
        output_ids.extend(outputs[:, inputs["input_ids"].size(1) :].tolist())
    return output_ids


def collate_fn(batch, tokenizer):
    prompts = [example["prompt"] for example in batch]
    inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    return inputs


def save_outputs(outputs, output_dir, process_index):
    output_file = os.path.join(output_dir, f"output_{process_index}.jsonl")
    with open(output_file, "w") as f:
        for line in outputs:
            f.write(json.dumps(line) + "\n")


def load_outputs(output_dir):
    outputs = []
    for file_name in os.listdir(output_dir):
        if file_name.startswith("output_") and file_name.endswith(".jsonl"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as f:
                for line in f:
                    outputs.append(json.loads(line.strip()))
            # os.remove(file_path)  # Delete the file after reading
    return outputs


if __name__ == "__main__":
    # Argument parsing for runtime configurations
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument("--dataset_path", type=str, default="BeIR/msmarco")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Specific model name")
    parser.add_argument("--save_path", type=str, default="data/msmarco_er.jsonl", help="Path to save inference results")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save intermediate outputs")
    parser.add_argument("--ner_opt", choices=["entityrag", "hipporag_original"], default="entityrag")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processors for processing datasets")
    parser.add_argument("--max_tokens", type=int, default=300, help="Generation config; max new tokens")
    parser.add_argument("--do_sample", type=bool, default=False, help="Generation config; sampling flag")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation config; temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Generation config; top k")
    parser.add_argument("--top_p", type=float, default=1.0, help="Generation config; top p, nucleus sampling")
    args = parser.parse_args()

    # Create the output directory if it doesn't already exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set the precision for floating-point matrix multiplications
    torch.set_float32_matmul_precision("high")

    # Initialize the accelerator
    accelerator = Accelerator()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model and tokenizer
    # Use flash attention for efficient inference
    dataset = load_dataset(args.dataset_path, "queries")["queries"].rename_column("text", "question")

    # Apply the appropriate NER option to map messages
    dataset = dataset.map(map_messages_hipporag) if args.ner_opt == "hipporag_original" else dataset.map(map_messages_entityrag)
    dataset = dataset.map(partial(build_prompt, tokenizer=tokenizer), num_proc=args.num_proc)

    # Prepare a DataLoader for efficient data batching
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_proc,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        pin_memory=True,
    )

    # Prepare the model and dataloader with the Accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    # Define the configuration for text generation
    generation_config = {
        "max_new_tokens": args.max_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    # Generate text outputs using the model
    output_ids = generate(model, tokenizer, dataloader, accelerator, generation_config)

    # Save the generated outputs to a file specific to the process
    save_outputs(output_ids, args.output_dir, accelerator.process_index)

    # Ensure all processes have completed saving their outputs
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Main process aggregates all outputs from the individual files
        output_ids = load_outputs(args.output_dir)

        # Add the outputs to the dataset and decode them
        dataset = dataset.add_column("output_ids", output_ids)
        dataset = dataset.map(partial(decode, tokenizer=tokenizer, feature="output"), num_proc=args.num_proc)

        # Apply post-processing to the results
        dataset = dataset.map(map_json_dict, num_proc=args.num_proc) if args.ner_opt == "hipporag_original" else dataset.map(map_entity, num_proc=args.num_proc)

        # Select only the relevant columns and save the final results
        dataset = dataset.select_columns(["_id", "title", "question", "entities"])
        dataset.to_json(args.save_path, orient="records", lines=True)

    # End the training or inference session
    accelerator.end_training()
