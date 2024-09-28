import os
from dataclasses import replace

from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .quality import RatingMetric

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
You are given a prompt and a response, and you provide a grade out of 100 measuring the quality of the response, in terms of accuracy, level of details, and typographical, grammatical and lexical correctness. 
Remove points as soon as one of the criteria is missed. <|eot_id|> 
<|start_header_id|>user<|end_header_id|> 
Prompt: {}\nResponse: {}<|eot_id|> <|start_header_id|>assistant<|end_header_id|> Grade: """
tokenizer_tokens = [
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "[/INST]",
    "[INST]",
    "<<SYS>>",
    "<</SYS>>",
    "<|im_start|>",
    "<|im_end|>",
]

class PPLRating(RatingMetric):
    def rate(self, generations, _):
        config = self.config
        writer_queue = self.writer_queue
        device = self.device

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        # Imports
        import torch

        from watermark_benchmark.utils import setup_randomness

        torch.set_num_threads(1)

        setup_randomness(config)

        # Setup server
        config.model = "meta-llama/Meta-Llama-3-8B-Instruct"
        config.max_new_tokens = 1
        config.dtype = "bfloat16"
        config.num_return_sequences = 1
        model = AutoModelForCausalLM.from_pretrained(config.model, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        tokenizer.pad_token = tokenizer.eos_token

        tasks = []
        for generation in generations:
            if "<<SYS>>" in generation.prompt:
                original_prompt = (
                    generation.prompt.split("<</SYS>>")[-1]
                    .replace("[INST]", "")
                    .replace("[/INST]", "")
                    .strip()
                )
                original_system_prompt = (
                    generation.prompt.split("<<SYS>>")[1]
                    .split("<</SYS>>")[0]
                    .strip()
                )
            elif "<|start_header_id|>system<|end_header_id|>" in generation.prompt:
                original_prompt = (
                    generation.prompt.split("<|start_header_id|>user<|end_header_id|>")[1]
                    .split("<|start_header_id|>assistant<|end_header_id|>")[0]
                    .strip()
                )
                original_system_prompt = (
                    generation.prompt.split("<|start_header_id|>system<|end_header_id|>")[1]
                    .split("<|start_header_id|>user<|end_header_id|>")[0]
                    .strip()
                )
            else:
                raise ValueError("Prompt format not recognized")

            original_response = generation.response

            full_prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
{original_system_prompt} <|eot_id|> <|start_header_id|>user<|end_header_id|>
{original_prompt} <|eot_id|> <|start_header_id|>assistant<|end_header_id|> 
{original_response} <|eot_id|>"""

            tasks.append(full_prompt)

        # Clip sequences that are too long
        max_token_length = 8000
        for i in tqdm(range(len(tasks)), total=len(tasks), desc="Encoding"):
            task = tasks[i]
            if len(task) > max_token_length:
                encoded_task = tokenizer(task)["input_ids"]
                if len(encoded_task) > max_token_length:
                    print(
                        "Warning: Task too long ({} tokens), clipping to {} tokens".format(
                            len(encoded_task), max_token_length
                        )
                    )
                    task = tokenizer.decode(encoded_task[:max_token_length])
            tasks[i] = task
        encodings = tokenizer(
            tasks,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_token_length,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(model.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]
        end_header_token_str = "<|end_header_id|>"
        end_header_token = tokenizer.encode(end_header_token_str)[-1]

        ppls = []
        batch_size = 16
        turn_index = 3
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]
            labels = encoded_batch

            with torch.no_grad():
                output = model(encoded_batch, attention_mask=attn_mask)
            logits = output.logits
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            shift_attention_mask_batch = attn_mask[..., 1:]
            for i in range(start_index, end_index):
                # Get the logits and labels for the specific example
                logits_example = shift_logits[i-start_index]
                encoded_example = shift_labels[i-start_index]
                attn_mask_example = shift_attention_mask_batch[i-start_index]
                # Find the token position of the 3rd end_header_token (which marks the start of the assistant response)
                end_of_nth_turn = [
                    idx for idx, token in enumerate(encoded_example) if token == end_header_token
                ]

                # We need the tokens after the third end_header_token, i.e., the assistant response
                if len(end_of_nth_turn) < turn_index:
                    print(f"Warning: Less than {turn_index} end_header_tokens found, skipping this example")
                    continue

                start_of_response = end_of_nth_turn[turn_index-1] + 1

                # Slice the logits and labels to only the assistant response
                response_logits = logits_example[start_of_response:-1].contiguous()  # Exclude the last token <|eot_id|>
                response_labels = encoded_example[start_of_response:-1].contiguous()
                response_attn_mask = attn_mask_example[start_of_response:-1].contiguous()

                # Compute the cross-entropy loss for these tokens
                ppls.append(torch.exp(
                    (loss_fct(response_logits, response_labels) * response_attn_mask).sum()
                    / response_attn_mask.sum()
                ))

        for i, generation in enumerate(generations):
            generations[i] = replace(generation, rating=ppls[i])

            # Write to file
        writer_queue.put(generations)

