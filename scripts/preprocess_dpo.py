from collections import defaultdict

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from watermark_benchmark import Generation

system_prompt = """
    You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences.\n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. Do not include any information that is not present in the original text. Do not respond with a greeting or any other extraneous information. Skip the preamble. Just rewrite the text directly.
"""
instruction = "\n[[START OF TEXT]]\n{}\n[[END OF TEXT]]"
response = "[[START OF PARAPHRASE]]\n"
response_fmt = "{}\n[[END OF PARAPHRASE]]"


def create_dpo_dataset(input_file: str, output_dir: str, model_family: str = "mistral") -> Dataset:
    generations = Generation.from_file(input_file)

    # Group generations by watermark and id
    grouped = {}
    for gen in generations:
        if gen.watermark is None:
            continue
        if gen.watermark.generator not in grouped:
            grouped[gen.watermark.generator] = defaultdict(list)
        grouped[gen.watermark.generator][gen.id].append(gen)
    all_data = []
    tokenizer = AutoTokenizer.from_pretrained(model_family)
    for watermark in grouped:
        dpo_data = []
        watermark_group = grouped[watermark]
        first = True
        for id in tqdm(watermark_group):
            id_group = watermark_group[id]
            no_attack = [g for g in id_group if g.attack is None]
            paraphrases = [g for g in id_group if g.attack is not None]

            if not no_attack or not paraphrases:
                continue

            if no_attack[0].pvalue > 0.01:
                continue

            valid_paraphrases = [g for g in paraphrases if g.rating > 0.8 and g.pvalue > 0.01]
            if not valid_paraphrases:
                continue

            chosen = max(valid_paraphrases, key=lambda g: g.pvalue)
            rejected = min(paraphrases, key=lambda g: g.pvalue)
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction.format(no_attack[0].response)}
            ],
                tokenize=False,
                add_generation_prompt=True
            )+response
            dpo_data.append({
                'prompt': prompt,
                'chosen': response_fmt.format(chosen.response),
                'rejected': response_fmt.format(rejected.response),
            })
            if first:
                print(f"Watermark: {watermark}")
                print(dpo_data[-1])
                first = False
        all_data.extend(dpo_data)
        dataset = Dataset.from_pandas(pd.DataFrame(dpo_data))
        dataset.save_to_disk(f"{output_dir}/{watermark}")
    dataset = Dataset.from_pandas(pd.DataFrame(all_data))
    dataset.save_to_disk(f"{output_dir}/all")
    print(f"Created DPO dataset with {len(all_data)} samples.")
    return dataset


# Usage
input_file = "run/results_dpo/detect.tsv"  # Replace with your input file name
output_dir = "dpo_dataset"  # Replace with your desired output directory
model_family = "mistralai/Mistral-7B-Instruct-v0.1"  # Replace with your model
dataset = create_dpo_dataset(input_file, output_dir, model_family)
