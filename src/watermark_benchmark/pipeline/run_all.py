import multiprocessing
import os
import shutil
import sys
from dataclasses import replace

from watermark_benchmark.pipeline.summarize import run as summary_run
from watermark_benchmark.utils import load_config
from watermark_benchmark.utils.classes import Generation, WatermarkSpec


def gen_wrapper(config, watermarks, custom_builder=None):
    config.baseline = True
    from watermark_benchmark.pipeline.generate import run as gen_run
    from datasets import load_dataset
    system_prompt =  "You are a helpful assistant. Always answer in the most accurate way."
    if config.prompt_dataset_hf is not None and len(config.prompt_dataset_hf) > 0:
        dataset = load_dataset(config.prompt_dataset_hf)
        prompts = dataset[config.prompt_dataset_split][config.prompt_dataset_column]
        prompts = prompts[:config.prompt_dataset_size]
        prompts = [(prompt, system_prompt) for prompt in prompts]
        gen_run(config, watermarks, custom_builder, raw_prompts=prompts)
    else:
        gen_run(config, watermarks, custom_builder)


def detect_wrapper(config, generations, custom_builder=None):
    from watermark_benchmark.pipeline.detect import run as detect_run

    detect_run(config, generations, custom_builder)


def perturb_wrapper(config, generations):
    from watermark_benchmark.pipeline.perturb import run as perturb_run

    perturb_run(config, generations)


def rate_wrapper(config, generations):
    from watermark_benchmark.pipeline.quality import run as rate_run

    rate_run(config, generations)


def run(
        config,
        watermarks,
        custom_builder=None,
        no_attack=False,
        GENERATE=True,
        PERTURB=True,
        RATE=True,
        DETECT=True,
):
    # Generation
    generations = []

    # Create output dir:
    try:
        os.mkdir(config.results)
    except Exception:
        pass

    print("### GENERATING ###")

    if GENERATE:
        config.input_file = None
        config.output_file = config.results + "/generations{}.tsv".format(
            "_val" if config.validation else ""
        )
        gen_wrapper(config, watermarks, custom_builder)

    print("### PERTURBING ###")

    # Perturb
    if PERTURB:
        config.input_file = config.results + "/generations{}.tsv".format(
            "_val" if config.validation else ""
        )
        config.output_file = config.results + "/perturbed{}.tsv".format(
            "_val" if config.validation else ""
        )
        if no_attack:
            shutil.copyfile(config.input_file, config.output_file)
        else:
            generations = Generation.from_file(config.input_file)
            perturb_wrapper(config, generations)

    print("### RATING ###")

    # Rate
    if RATE:
        config.input_file = config.results + "/perturbed{}.tsv".format(
            "_val" if config.validation else ""
        )
        config.output_file = config.results + "/rated{}.tsv".format(
            "_val" if config.validation else ""
        )
        generations = Generation.from_file(config.input_file)
        rate_wrapper(config, generations)

    print("### DETECTING ###")

    # Detect
    if DETECT:
        config.input_file = config.results + "/rated{}.tsv".format(
            "_val" if config.validation else ""
        )
        config.output_file = config.results + "/detect{}.tsv".format(
            "_val" if config.validation else ""
        )
        generations = Generation.from_file(config.input_file)
        detect_wrapper(config, generations, custom_builder)
        generations = Generation.from_file(config.output_file)
    else:
        generations = Generation.from_file(
            config.results
            + "/detect{}.tsv".format("_val" if config.validation else "")
        )

    return generations


def main():
    multiprocessing.set_start_method("spawn")
    config = load_config(sys.argv[1])
    with open(config.watermark, encoding="utf-8") as infile:
        watermarks = [
            replace(WatermarkSpec.from_str(l.strip()), tokenizer=config.model)
            for l in infile.read().split("\n")
            if len(l)
        ]
    generations = run(config, watermarks)

    summary_run(config, generations)


def full_pipeline(
        config_file,
        watermarks,
        custom_builder=None,
        run_validation=False,
        no_attack=False,
):
    multiprocessing.set_start_method("spawn")
    config = (
        load_config(config_file)
        if isinstance(config_file, str)
        else config_file
    )
    if isinstance(watermarks, str):
        with open(config.watermark, encoding="utf-8") as infile:
            watermarks = [
                replace(
                    WatermarkSpec.from_str(line.strip()), tokenizer=config.model
                )
                for line in infile.read().split("\n")
                if len(line)
            ]

    generations = run(config, watermarks, custom_builder, no_attack=no_attack)

    if not run_validation:
        return summary_run(config, generations)

    _, _, validation_watermarks = summary_run(config, generations)

    # Validation
    print("#### STARTING VALIDATION ####")
    config.validation = True
    generations = run(config, validation_watermarks, no_attack=no_attack)

    return summary_run(config, generations)


if __name__ == '__main__':
    main()
