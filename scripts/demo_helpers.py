import multiprocessing
import os
import random
from collections import defaultdict
from dataclasses import replace

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from watermark_benchmark.servers import get_model
from watermark_benchmark.utils import get_server_args
from watermark_benchmark.utils import setup_randomness
from watermark_benchmark.utils.apis import custom_model_process
from watermark_benchmark.utils.apis import llm_rating_process
from watermark_benchmark.watermark import get_watermark


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
                    target=custom_model_process,
                    args=(custom_model_queues[j], config.custom_model_paths[j], [devices[j%len(devices)]], config),
                )
            )
            paraphrase_processes[-1].start()

        return dispatch_queues, response_queues, paraphrase_processes

    def paraphrase(self, samples):
        paraphrased_samples = defaultdict(list)
        for id in samples:
            for sample in samples[id]:
                for custom_path in self.config.custom_model_paths:
                    self.dispatch_queues[custom_path].put((sample.response, self.response_queues[custom_path]))
        pbar = tqdm(total=len(samples)*len(samples[0])*len(self.config.custom_model_paths))
        for custom_path in self.config.custom_model_paths:
            for id in samples:
                for i in range(len(samples[id])):
                    response = self.response_queues[custom_path].get(block=True)
                    paraphrased_samples[id].append(
                        replace(samples[id][i], response=response[0], attack=custom_path)
                    )
                    pbar.update(1)
        return paraphrased_samples

    def kill(self):
        for queue in self.dispatch_queues.values():
            queue.put(None)
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
            for id in sample_list:
                self.rating_queue.put((sample_list[id], self.response_queue))

        for sample_list in sample_lists:
            for _ in tqdm(range(len(sample_list))):
                response = self.response_queue.get(block=True)
                id = response[0].id
                sample_list[id] = response

        return sample_lists

    def kill(self):
        self.rating_queue.put(None)
        for process in self.processes:
            process.terminate()
