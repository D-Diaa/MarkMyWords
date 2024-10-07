import copy
import gc
import json
import multiprocessing
import os
import random
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List

import gradio as gr
import torch
from gradio import update
from math import ceil
from tqdm import tqdm
from transformers import AutoTokenizer

from src.watermark_benchmark import ConfigSpec
from src.watermark_benchmark.servers import get_model
from src.watermark_benchmark.utils import get_server_args, get_tokenizer
from src.watermark_benchmark.utils import setup_randomness
from src.watermark_benchmark.utils.apis import llm_rating_process
from src.watermark_benchmark.watermark import get_watermark
from watermark_benchmark import Generation
from watermark_benchmark.metrics.llm_rating import tokenizer_tokens


def get_samples(path = "/home/ubuntu/repos/MarkMyWords/fix_rjtp/results/generations.tsv"):
    all_samples = Generation.from_file(path)
    watermarked_samples, baseline_samples = defaultdict(list), defaultdict(list)
    for sample in all_samples:
        if sample.watermark is not None:
            if sample.watermark.generator in ["distributionshift"]:
                watermarked_samples[sample.id].append(sample)
        else:
            baseline_samples[sample.id].append(sample)
    return watermarked_samples

class TextProcessor:
    def __init__(self, paraphraser, detector, rater, all_samples):
        self.paraphraser = paraphraser
        self.detector = detector
        self.rater = rater
        self.all_samples = all_samples
        self.current_samples = None
        self.paraphrased_samples = None
        self.get_new_samples()  # Initialize with first set

    def get_new_samples(self):
        """Selects and processes a new random subset of samples."""
        random_keys = random.sample(list(self.all_samples.keys()), 5)
        self.current_samples = {
            k: self.all_samples[k]
            for k in random_keys
        }
        self.current_samples = self.detector.detect(self.current_samples)
        self.current_samples = self.rater.rate([self.current_samples])[0]
        self.paraphrased_samples = self.current_samples


    def colorize_tokens(self, sample) -> str:
        """Colorizes tokens in the text based on their assigned colors."""
        if not sample.colors:
            return self._wrap_text_in_container(sample.response)

        result = []
        prev_idx = 0

        for token, color in sample.colors:
            found_idx = sample.response.find(token, prev_idx)
            if found_idx == -1:
                continue

            # Add text before token
            result.append(sample.response[prev_idx:found_idx])

            # Add colored token
            color_code = "rgb(255, 150, 150)" if color == 0 else "rgb(150, 255, 150)"
            result.append(
                f"<span style='color:{color_code}'>{token}</span>"
            )

            prev_idx = found_idx + len(token)

        # Add remaining text
        result.append(sample.response[prev_idx:])
        return self._wrap_text_in_container(''.join(result))
    def _wrap_metrics_in_container(self, pvalue: float, rating: float, delta:float) -> str:
        """Wraps pvalue and rating in a styled container."""
        return f"""
            <div style='
                display: flex;
                justify-content: space-between;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 8px 16px;
                margin: 8px 0;
                font-family: system-ui, -apple-system, sans-serif;
            '>
                <span><strong>PValue:</strong> {pvalue:.4f}</span>
                <span><strong>Beta (Strength):</strong> {delta:.4f}</span>
                <span><strong>LLM Rating:</strong> {rating:.2f}</span>
            </div>
        """
    def _wrap_text_in_container(self, text: str) -> str:
        """Wraps text in a styled container."""
        return f"""
            <div style='
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 16px;
                margin: 8px 0;
                font-family: system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                white-space: pre-wrap;
                overflow-x: auto;
            '>
                {text.replace("[[END OF PARAPHRASE]]", "")}
            </div>
        """

    def display_colored_text(self, samples: Dict[str, List]) -> str:
        """Displays all samples with proper formatting."""
        colored_texts = []
        # print(samples)
        samples = sorted(samples.items())
        for id, sample_list in samples:
            cleaned_prompt = sample_list[0].prompt
            for token in tokenizer_tokens:
                cleaned_prompt = cleaned_prompt.replace(token, "").strip()
            print(cleaned_prompt)
            colored_texts.append(f"<h4>{cleaned_prompt}</h4>")
            for sample in sample_list:
                colored_texts.append(self.colorize_tokens(sample))
                print(sample.pvalue, sample.rating)
                colored_texts.append(self._wrap_metrics_in_container(sample.pvalue, sample.rating, sample.watermark.delta))

        return "\n".join(colored_texts)

    def process_samples(self) -> str:
        """Processes current samples through detection and returns table HTML."""
        print("Processing samples...")
        self.paraphrased_samples = self.paraphraser.paraphrase(self.current_samples.copy())
        self.paraphrased_samples = self.detector.detect(self.paraphrased_samples)
        self.paraphrased_samples = self.rater.rate([self.paraphrased_samples])[0]
        return self.display_samples_table(self.current_samples, self.paraphrased_samples)


    def paraphrase_user_text(self, text: str) -> str:
        """Paraphrases user text without detection."""
        if not text.strip():
            return ""
        print(f"Paraphrasing user text: {text}")
        # Create a temporary sample with user text
        sample_id = list(self.current_samples.keys())[0]
        temp_samples = {
            0: [
                replace(self.current_samples[sample_id][0], response=text, watermark=None, id=0)
            ]
        }

        # Only paraphrase, don't detect
        temp_samples = self.paraphraser.paraphrase(temp_samples)

        # Return the paraphrased text in a styled container
        return self._wrap_text_in_container(
            temp_samples[0][0].response
        )

    def display_samples_table(self, original_samples, processed_samples):
        table_rows = []
        all_ids = set(original_samples.keys()) | set(processed_samples.keys())

        for id in sorted(all_ids):
            original = original_samples.get(id, [])
            processed = processed_samples.get(id, [])

            original_html = self.display_colored_text({id: original}) if original else ""
            processed_html = self.display_colored_text({id: processed}) if processed else ""

            table_rows.append(f"""
                <tr>
                    <td style="vertical-align: top; width: 50%;">{original_html}</td>
                    <td style="vertical-align: top; width: 50%;">{processed_html}</td>
                </tr>
            """)

        table_html = f"""
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th style="width: 50%; text-align: left; padding: 10px; ">Original Samples</th>
                    <th style="width: 50%; text-align: left; padding: 10px; ">Processed Samples</th>
                </tr>
                {"".join(table_rows)}
            </table>
        """
        return table_html

def get_paraphraser_selection():
    file_path = "paraphraser_selection.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f).get("paraphraser")
    return None

def save_paraphraser_selection(paraphraser):
    with open("paraphraser_selection.json", "w") as f:
        json.dump({"paraphraser": paraphraser}, f)
def create_interface(processor: TextProcessor):
    """Creates and configures the Gradio interface."""
    paraphrasers_dir = "/home/ubuntu/repos/MarkMyWords/"  # Update this path
    paraphraser_files = [f for f in os.listdir(paraphrasers_dir) if f.startswith('dpo') and "dataset" not in f] + ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-2-7b-chat-hf']

    with gr.Blocks(css="""
        .container { margin: 15px; }
        .sample-box { margin: 10px 0; }
        .user-input-section { margin-top: 30px; border-top: 2px solid #eee; padding-top: 20px; }
    """) as demo:
        gr.Markdown("""
        # Adaptive Paraphraser Demo
        - This demo showcases the adaptive paraphraser with DistributionShift watermark detection.
        - Paper: [Optimizing Adaptive Attacks Against Content Watermarks for Language Models](https://arxiv.org/pdf/2410.02440)
        - Contact: [Abdulrahman Diaa](https://d-diaa.github.io)
        """)
        # horizontal line
        gr.HTML("<hr>")
        # Paraphraser selection section
        gr.Markdown("## Paraphraser Selection")
        paraphraser_state = gr.State(get_paraphraser_selection())

        def refresh_paraphraser():
            current_paraphraser = get_paraphraser_selection()
            return current_paraphraser, f"{current_paraphraser} loaded"
        if paraphraser_state.value is None:
            paraphraser_state.value = processor.paraphraser.config.custom_model_paths[0].replace(paraphrasers_dir,"") if processor.paraphraser.config.custom_model_paths else None

        with gr.Row():
            with gr.Column():
                paraphraser_dropdown = gr.Dropdown(
                    choices=paraphraser_files,
                    label="Select Paraphraser",
                    value=paraphraser_state.value,
                )
            with gr.Column():
                status = gr.Textbox(label="Paraphraser Status", value=f"{paraphraser_state.value} loaded", lines=1)

        with gr.Row():
            load_paraphraser_btn = gr.Button("Load Paraphraser")
        gr.HTML("<hr>")

        gr.Markdown("## Paraphrase your own text")
        # User input section
        with gr.Row():
            user_input = gr.Textbox(
                label="Try Your Own Text",
                placeholder="Type or paste your text here...",
                lines=2
            )
        with gr.Row():
            paraphrase_btn = gr.Button("Paraphrase", variant="primary")

        with gr.Row():
            user_output = gr.HTML(label="Paraphrased Result")
        gr.HTML("<hr>")

        gr.Markdown("""
        ## Try watermarked samples
        - 🟢 Green text indicate a green token (Watermark Detected) in DistributionShift Watermark
        - 🔴 Red text indicate a red token (Watermark NOT Detected) in DistributionShift Watermark
        """)
        with gr.Row():
            change_samples_btn = gr.Button("Get New Samples", variant="secondary")
            detect_btn = gr.Button("Paraphrase and Detect", variant="primary")

        # Sample comparison section
        samples_table = gr.HTML(
            processor.display_samples_table(processor.current_samples, {}),
            elem_classes=["sample-box"]
        )


        # Event handlers
        def update_samples():
            print("Updating samples...")
            processor.get_new_samples()
            return processor.display_samples_table(processor.current_samples, {})
        def load_paraphraser(paraphraser_file):
            global loaded_paraphraser
            loaded_paraphraser = paraphraser_file
            save_paraphraser_selection(paraphraser_file)

            config_copy  = copy.deepcopy(processor.paraphraser.config)
            global_manager = processor.paraphraser.global_manager
            devices = processor.paraphraser.devices
            model_path = os.path.join(paraphrasers_dir, paraphraser_file) if "dpo" in paraphraser_file else paraphraser_file
            config_copy.custom_model_paths = [model_path]
            print(f"Loading paraphraser: {paraphraser_file}")
            yield update(interactive=False), update(interactive=False), f"Clearing old paraphraser..."
            processor.paraphraser.kill()
            del processor.paraphraser
            gc.collect()
            torch.cuda.empty_cache()
            processor.paraphraser = Paraphraser(config_copy, global_manager, devices)
            yield update(interactive=False), update(interactive=False), f"Loading new paraphraser: {paraphraser_file}..."
            sample_id = list(processor.current_samples.keys())[0]
            temp_samples = {
                0: [
                    replace(processor.current_samples[sample_id][0], response="This is a test", watermark=None, id=0)
                ]
            }
            _ = processor.paraphraser.paraphrase(temp_samples)

            yield update(interactive=True), update(interactive=True), f"{paraphraser_file} loaded successfully."
        change_samples_btn.click(
            fn=update_samples,
            outputs=[samples_table]
        )

        detect_btn.click(
            fn=processor.process_samples,
            outputs=[samples_table]
        )

        paraphrase_btn.click(
            fn=processor.paraphrase_user_text,
            inputs=[user_input],
            outputs=[user_output]
        )
        demo.load(fn=refresh_paraphraser, outputs=[paraphraser_dropdown, status])

        load_paraphraser_btn.click(
            fn=load_paraphraser,
            inputs=[paraphraser_dropdown],
            outputs=[paraphrase_btn, detect_btn, status],
        ).then(
            None,
            None,
            None,
            js = "() => {window.location.reload();}"
        )
    return demo




def prepare_prompts(model, raw_prompts, watermarks, temp):
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [tokenizer.apply_chat_template([
        {
            "role": "system",
            "content": s,
        },
        {
            "role": "user",
            "content": p,
        },
    ], tokenize=False, add_generation_prompt=True) for p, s in raw_prompts]
    tasks = []
    for watermark in watermarks:
        keys = [random.randint(0, 1000000) for _ in prompts]
        tasks.append((watermark, keys))
    all_tasks = [(watermark, keys, watermark.temp) for watermark, keys in tasks]
    all_tasks.append((None, None, temp))
    return all_tasks, prompts


class Detector:
    def __init__(self, config: ConfigSpec, global_manager):
        self.config = config
        self.global_manager = global_manager
        self.dispatch_queue = self.global_manager.Queue()
        self.response_queue = self.global_manager.Queue()
        self.processes = []

        for device in config.devices:
            for _ in range(config.detections_per_gpu):
                process = multiprocessing.Process(
                    target=Detector.run,
                    args=(config, self.dispatch_queue, self.response_queue, device)
                )
                process.start()
                self.processes.append(process)
    @staticmethod
    def run(config, dispatch_queue, response_queue, device):
        # Setup device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(1)

        # Randomness and models
        tokenizer = get_tokenizer(config.model)

        while True:
            generations = dispatch_queue.get()
            if generations is None:
                break
            setup_randomness(config)
            watermark = generations[0].watermark
            full = []
            unique_keys, keys, key_indices = {}, [], []
            for g in generations:
                if g.key not in unique_keys:
                    unique_keys[g.key] = len(keys)
                    keys.append(g.key)
                key_indices.append(unique_keys[g.key])

            watermark_engine = get_watermark(
                watermark,
                tokenizer,
                None,
                [0],
                keys,
                None,
            )
            return_cumul = watermark.generator == "distributionshift"
            for g_idx, g in tqdm(enumerate(generations), total=len(generations)):
                verifier_outputs = watermark_engine.verify_text(
                    g.response,
                    exact=True,
                    index=key_indices[g_idx],
                    meta={"prompt": g.prompt},
                    return_cumul=return_cumul,
                )
                sep_watermarks = watermark.sep_verifiers()

                for verifier_index, verifier_output in enumerate(
                        verifier_outputs.values()
                ):
                    cumul = []
                    if return_cumul:
                        verifier_output, cumul = verifier_output
                    tokens = watermark_engine.tokenizer.encode(g.response, add_special_tokens=False, return_tensors="pt")
                    # decode every token separately
                    decoded = [watermark_engine.tokenizer.decode(t) for t in tokens.squeeze().tolist()]
                    pvalue = verifier_output.get_pvalue()
                    eff = verifier_output.get_size(watermark.pvalue)
                    full.append(
                        replace(
                            g,
                            watermark=replace(
                                sep_watermarks[verifier_index], secret_key=g.key
                            ),
                            pvalue=pvalue,
                            efficiency=eff,
                            colors=[(decoded[i], cumul[i]) for i in range(len(cumul))],
                        )
                    )

            response_queue.put(full)

    def detect(self, generation_dicts):
        generations = sum(generation_dicts.values(), start=[])
        generations_per_watermark = defaultdict(list)
        num_processes = 0
        for g in generations:
            generations_per_watermark[str(g.watermark)].append(g)
        for watermark in generations_per_watermark:
            generations = generations_per_watermark[watermark]
            tasks_per_process = ceil(len(generations) / len(self.processes))
            num_processes +=ceil(len(generations) / tasks_per_process)
            for i in range(0, len(generations), tasks_per_process):
                self.dispatch_queue.put(generations[i:i + tasks_per_process])

        all_generations = []
        for _ in tqdm(range(num_processes)):
            all_generations.extend(self.response_queue.get())
        result = defaultdict(list)
        for generation in all_generations:
            result[generation.id].append(generation)
        return result

    def terminate(self):
        for process in self.processes:
            process.terminate()


class WatermarkGenerator:
    def __init__(self, config, global_manager, devices):
        self.config = config
        self.devices = devices
        self.global_manager = global_manager
        self.dispatch_queue, self.response_queue, self.processes = self._prepare_generator()

    @staticmethod
    def _initialize_watermark_server(config, devices):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
        engine = "vllm" if len(devices) > 1 else "hf"
        return get_model(
            engine, config, max_model_len=2048, **get_server_args(config)
        )

    def _prepare_generator(self):
        dispatch_queue = self.global_manager.Queue()
        response_queue = self.global_manager.Queue()
        processes = []
        process = multiprocessing.Process(
            target=WatermarkGenerator._generation_process,
            args=(self.config, dispatch_queue, response_queue, self.devices)
        )
        process.start()
        processes.append(process)
        return dispatch_queue, response_queue, processes

    @staticmethod
    def _generation_process(config, dispatch_queue, response_queue, devices):
        watermark_server = WatermarkGenerator._initialize_watermark_server(config, devices)
        while True:
            task = dispatch_queue.get()
            if task is None:
                break
            watermark, keys, temp, prompts = task
            setup_randomness(config)
            watermark_engine = (
                get_watermark(
                    watermark,
                    watermark_server.tokenizer(),
                    None,
                    watermark_server.devices,
                    keys,
                    builder=None,
                )
                if watermark is not None
                else None
            )

            watermark_server.install(watermark_engine)
            outputs = watermark_server.run(prompts, config, temp, keys, watermark, use_tqdm=False)
            watermark_server.install(None)
            torch.cuda.empty_cache()
            response_queue.put(outputs)

    def generate(self, prompts, tasks):
        watermarked_samples = defaultdict(list)
        baseline_samples = defaultdict(list)
        for task in tasks:
            self.dispatch_queue.put((*task, prompts))

        for _ in tqdm(tasks):
            sample_list = self.response_queue.get()
            for sample in sample_list:
                if sample.watermark is None:
                    baseline_samples[sample.id].append(sample)
                else:
                    watermarked_samples[sample.id].append(sample)

        return watermarked_samples, baseline_samples

    def kill(self):
        for _ in self.config.devices:
            self.dispatch_queue.put(None)
        for process in self.processes:
            process.join()
        torch.cuda.empty_cache()


class Paraphraser:
    def __init__(self, config, global_manager, devices):
        self.config = config
        self.devices = devices
        self.global_manager = global_manager
        self.dispatch_queues, self.response_queues, self.processes = self._prepare_paraphrasers(config, global_manager)

    def _prepare_paraphrasers(self, config, global_manager):
        devices = self.devices
        custom_model_queues = [global_manager.Queue() for _ in range(len(config.custom_model_paths))]
        dispatch_queues = {f"{config.custom_model_paths[i]}": custom_model_queues[i] for i in
                           range(len(config.custom_model_paths))}
        response_queues = {f"{config.custom_model_paths[i]}": global_manager.Queue() for i in
                           range(len(config.custom_model_paths))}

        paraphrase_processes = []
        for j in range(len(config.custom_model_paths)):
            paraphrase_processes.append(
                multiprocessing.Process(
                    target=Paraphraser.paraphrase_process,
                    args=(custom_model_queues[j], config.custom_model_paths[j], [devices[j % len(devices)]], config),
                )
            )
            paraphrase_processes[-1].start()

        return dispatch_queues, response_queues, paraphrase_processes

    @staticmethod
    def paraphrase_process(queue, model, devices, config):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in devices)
        # device = "cuda" if len(devices) else "cpu"
        model_kwargs = get_server_args(config)
        model_kwargs[
            "max_model_len"] = config.max_new_tokens + config.custom_max_new_tokens + 512  # 512 is a buffer for system prompt
        adapter_path = None
        if model is not None and os.path.exists(
                os.path.join(model, "adapter_config.json")) and not os.path.exists(
                os.path.join(model, "config.json")):
            print(f"Loading base and then generating from adapter '{model}'")
            adapter_path = model
            adapter_config = json.load(open(os.path.join(model, "adapter_config.json")))
            base_path = adapter_config.setdefault("base_model_name_or_path", "?")
            model = base_path
        lora_request = None
        if adapter_path is not None:
            lora_request = LoRARequest(adapter_path, devices[0] + 1, adapter_path)
        torch.cuda.empty_cache()
        server = LLM(model, enable_lora=(adapter_path is not None), max_lora_rank=64, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model)
        gen_params = SamplingParams(
            temperature=config.custom_temperature,
            max_tokens=config.custom_max_new_tokens,
            n=config.custom_batch,
        )
        system_prompt = (
            "You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences.\n"
            "Ensure that the final output contains the same information as the original text and has roughly the same length.\n"
            "Do not leave out any important details when rewriting in your own voice. Do not include any information that is not"
            "present in the original text. Do not respond with a greeting or any other extraneous information. "
            "Skip the preamble. Just rewrite the text directly."
        )
        instruction = "\n[[START OF TEXT]]\n{}\n[[END OF TEXT]]"
        response = "[[START OF PARAPHRASE]]\n"

        while True:
            task = queue.get(block=True)
            if task is None:
                return
            generations, destination_queue = task
            prompts = [tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction.format(g.response)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            ) + response for g in generations]
            outputs = server.generate(prompts, gen_params, use_tqdm=True, lora_request=lora_request)
            paraphrased = []
            for i, output in enumerate(outputs):
                paraphrased_text = output.outputs[0].text.strip().replace("[[END OF PARAPHRASE]]", "").replace('[END OF PARAPHRASE]', '')
                paraphrased.append(replace(generations[i], response=paraphrased_text))
            destination_queue.put(paraphrased)
            torch.cuda.empty_cache()

    def paraphrase(self, samples):
        samples_list = sum(samples.values(), start=[])
        paraphrased_samples = defaultdict(list)
        for custom_path in self.config.custom_model_paths:
            self.dispatch_queues[custom_path].put((samples_list, self.response_queues[custom_path]))
        for custom_path in self.config.custom_model_paths:
            paraphrased_list = self.response_queues[custom_path].get(block=True)
            for g in paraphrased_list:
                paraphrased_samples[g.id].append(g)
        return paraphrased_samples

    def kill(self):
        for queue in self.dispatch_queues.values():
            queue.put(None)
        for process in self.processes:
            process.join()
        torch.cuda.empty_cache()
        for process in self.processes:
            process.terminate()


class Rater:
    def __init__(self, config, global_manager, device):
        self.device = device
        self.rating_queue, self.response_queue, self.processes = self._prepare_rater(config, global_manager)

    def _prepare_rater(self, config, global_manager):
        rating_queue = global_manager.Queue()
        response_queue = global_manager.Queue()

        rating_process = multiprocessing.Process(
            target=llm_rating_process,
            args=(rating_queue, [self.device], config),
        )
        rating_process.start()

        return rating_queue, response_queue, [rating_process]

    def rate(self, sample_lists):
        for sample_list in sample_lists:
            generations = sum(sample_list.values(), start=[])

            self.rating_queue.put((generations, self.response_queue))
        new_lists = []
        for _ in range(len(sample_lists)):
            generations = self.response_queue.get(block=True)
            new_lists.append(defaultdict(list))
            for generation in generations:
                id = generation.id
                new_lists[-1][id].append(generation)

        return new_lists

    def kill(self):
        self.rating_queue.put(None)
        for process in self.processes:
            process.terminate()
