import multiprocessing
from scripts.demo_helpers import Detector, Paraphraser, Rater, TextProcessor, create_interface, get_samples, \
    get_paraphraser_selection
from src.watermark_benchmark import ConfigSpec

# multiprocessing.set_start_method("spawn")
global_manager = multiprocessing.Manager()
paraphrases_folder = '/home/ubuntu/repos/MarkMyWords'
paraphrase_models = [get_paraphraser_selection()]
if paraphrase_models[0] is None:
    paraphrase_models = ["meta-llama/Meta-Llama-3-8B-Instruct"]
else:
    paraphrase_models = [f"{paraphrases_folder}/{paraphrase_models[0]}"]
devices = ['2','3']
config = ConfigSpec(
    num_return_sequences= 1,
    engine= "vllm",
    baseline = True,
    watermark = "watermark_specs",
    max_new_tokens = 512,
    seed=42,
    paraphrase = True,
    custom_processes = 1,
    translate_processes = 1,
    custom_only = True,
    threads = 32,
    misspellings = "static_data/misspellings.json",
    devices = devices,
    detections_per_gpu = 4,
    quality_metric = "llm",
    gpu_memory_utilization = 0.85,
    dtype = "bfloat16",
    custom_model_paths = paraphrase_models,
    custom_batch = 1,
    custom_temperature = 1.0,
    custom_max_new_tokens = 512,
)

detector = Detector(config, global_manager)
paraphraser = Paraphraser(config, global_manager, [int(devices[0])])
rater = Rater(config, global_manager, devices[1])
watermarked_samples = get_samples()
processor = TextProcessor(paraphraser, detector, rater, watermarked_samples)
demo = create_interface(processor)
demo.launch(share=True)