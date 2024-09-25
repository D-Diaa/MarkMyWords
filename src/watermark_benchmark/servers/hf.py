""" VLLM Server """

from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
)

from watermark_benchmark.utils.classes import (
    ConfigSpec,
    Generation,
    WatermarkSpec,
)
from watermark_benchmark.utils.stats import Stats
from .server import Server


class HFServer(Server, LogitsProcessor):
    """
    A Hugging Face based watermarking server
    """

    def __init__(self, config: Dict[str, Any], **kwargs) -> None:
        """
        Initializes the HF server.

        Args:
        - config (Dict[str, Any]): A dictionary containing the configuration of the model.
        - **kwargs: Additional keyword arguments.
        """
        model = config.model
        self.server = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        self.devices = [[i for i in range(torch.cuda.device_count())][0]]
        self.watermark_engine = None
        self.batch_size = config.hf_batch_size
        self.current_batch = 0
        self.current_offset = 0

    def tokenizer(self):
        """
        Returns the tokenizer.
        """
        return self._tokenizer
    def install(self, watermark_engine) -> None:
        """
        Installs the watermark engine.

        Args:
        - watermark_engine (Any): The watermark engine.
        """
        self.watermark_engine = watermark_engine

    def __call__(self, input_ids, scores):
        # Apply watermarking
        ids = [
            self.current_offset + (self.current_batch * self.batch_size) + i
            for i in range(input_ids.shape[0])
        ]
        self.stats.update(scores, ids)
        if self.watermark_engine is not None:
            scores = self.watermark_engine.process(scores, input_ids, ids)
        return scores

    def run(
            self,
            inputs: List[str],
            config: ConfigSpec,
            temp: float,
            keys: Optional[List[int]] = None,
            watermark_spec: Optional[WatermarkSpec] = None,
            use_tqdm=False,
            **kwargs,
    ) -> List[Generation]:
        """
        Runs the server.

        Args:
        - inputs (List[str]): A list of input strings.
        - config (ConfigSpec): The configuration.
        - temp (float): The temperature.
        - keys (Optional[List[int]]): A list of keys.
        - watermark_spec (Optional[WatermarkSpec]): The watermark specification.
        - use_tqdm (bool): A boolean indicating whether to use tqdm.

        Returns:
        - List[Generation]: A list of generations.
        """
        # Setup logit processor
        processors = LogitsProcessorList()
        processors.append(self)

        # Run
        generations = []
        self.stats = Stats(len(inputs), temp)
        while True:
            try:
                self.current_offset = len(generations)
                for batch_start in tqdm(
                        range(0, len(inputs) - len(generations), self.batch_size),
                        total=(len(inputs) - len(generations)) // self.batch_size,
                        # description=f"Generating text (batch size {self.batch_size})",
                        disable=not use_tqdm,
                ):
                    prompts = inputs[self.current_offset+batch_start: self.current_offset+ batch_start+self.batch_size]
                    self.current_batch = batch_start//self.batch_size
                    batch = self._tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                    ).input_ids.to(self.server.device)
                    outputs = self.server.generate(
                        batch,
                        temperature=temp,
                        max_new_tokens=config.max_new_tokens,
                        num_return_sequences=config.num_return_sequences,
                        do_sample=(temp > 0),
                        logits_processor=processors,
                    )
                    responses = self._tokenizer.batch_decode(outputs[:, batch[0].shape[0]:], skip_special_tokens=True)
                    generations.extend(
                        [
                            Generation(
                                (
                                    watermark_spec
                                    if watermark_spec is not None
                                    else None
                                ),
                                (
                                    keys[
                                        self.current_offset
                                        + batch_start
                                        + j
                                        ]
                                    if keys is not None
                                    else None
                                ),
                                None,
                                self.current_offset + batch_start + j,
                                prompts[j],
                                responses[j],
                                None,
                                None,
                                None,
                                *self.stats[
                                    self.current_offset
                                    + batch_start
                                    + j
                                    ],
                                temp,
                                [],
                            )
                            for j in range(len(responses))
                        ]
                    )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.batch_size > 1:
                    torch.cuda.empty_cache()
                    self.batch_size = self.batch_size // 2
                    continue
                else:
                    raise e
            break

        self.current_batch = 0
        self.current_offset = 0
        return generations

