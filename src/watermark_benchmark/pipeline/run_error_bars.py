import multiprocessing
import os
import sys
from dataclasses import replace

from watermark_benchmark.pipeline.summarize import run as summary_run
from watermark_benchmark.utils import load_config
from watermark_benchmark.utils.classes import Generation, WatermarkSpec


def gen_wrapper(config, watermarks, custom_builder=None):
    config.baseline = True
    from watermark_benchmark.pipeline.generate import run as gen_run

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

    print("### DETECTING ###")

    # Detect
    if DETECT:
        config.input_file = config.results + "/generations{}.tsv".format(
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
    iteration_count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    with open(config.watermark, encoding="utf-8") as infile:
        watermarks = [
            replace(WatermarkSpec.from_str(l.strip()), tokenizer=config.model)
            for l in infile.read().split("\n")
            if len(l)
        ]

    orignal_results_path = config.results
    for iteration in range(iteration_count):
        config.results = os.path.join(
            orignal_results_path, f"iteration_{iteration}"
        )
        os.makedirs(config.results, exist_ok=True)
        config.seed = iteration
        run(config, watermarks)


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
