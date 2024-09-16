import os
import time

import math
import tiktoken
import torch
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from watermark_benchmark.utils import get_server_args

openai_cache = {}

_MAX_TOKENS_BY_MODEL = {
    "text-babbage-001": 2048,
    "text-curie-001": 2048,
    "text-davinci-001": 2048,
    "text-davinci-002": 4096,
    "text-davinci-003": 4096,
}


### APIS ###


def call_openai(
        model,
        temperature,
        top_p,
        presence_penalty,
        frequency_penalty,
        prompt,
        system_prompt,
        max_tokens,
        timeout,
        logprobs,
        echo,
        client=None,
        cache=openai_cache,
):
    # Call openai API and return results

    if client is None:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    serialization = str(
        (
            model,
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            prompt,
            system_prompt,
            max_tokens,
            timeout,
            logprobs,
        )
    )

    if cache is not None and temperature == 0 and serialization in cache:
        return cache[serialization]

    def loop(f, params):
        retry = 0
        while retry < 5:
            try:
                return f(params)
            except Exception as e:
                if retry > 2:
                    print(f"Error {retry}: {e}")
                if "Rate limit" in str(e) or "overloaded" in str(e):
                    time.sleep(45)
                else:
                    time.sleep(3 * retry)
                params["timeout"] += 3
                retry += 1
                continue
        return None

    is_chat = "gpt" in model
    request_params = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if presence_penalty is not None:
        request_params["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        request_params["frequency_penalty"] = frequency_penalty
    if logprobs is not None:
        request_params["logprobs"] = logprobs

    if is_chat:
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]

        request_params["messages"] = messages
        completion = loop(
            lambda x: client.chat.completions.create(**x), request_params
        )
        if temperature == 0 and cache is not None:
            cache[serialization] = completion

        return completion
    else:
        request_params["echo"] = echo
        encoder = tiktoken.encoding_for_model(model)
        request_params["prompt"] = encoder.decode(
            encoder.encode(prompt)[: _MAX_TOKENS_BY_MODEL[model]]
        )
        completion = loop(
            lambda x: client.completions.create(**x), request_params
        )

        if temperature == 0 and cache is not None:
            cache[serialization] = completion

        return completion


def call_dipper(model, tokenizer, texts, device="cpu"):
    input = {
        k: v.to(device) if type(v) == torch.Tensor else v
        for k, v in tokenizer.batch_encode_plus(
            texts, padding=True, return_tensors="pt"
        ).items()
    }
    orig_size = input["input_ids"].shape[0]
    batch_size = 1 << (orig_size - 1).bit_length()

    text_output = ""
    try:
        while True:
            if orig_size > batch_size:
                rounds = math.ceil(orig_size / batch_size)
                round_size = batch_size
            else:
                rounds = 1
                round_size = orig_size

            text_output = ""
            error = False

            for r in range(rounds):
                truncated_input = {
                    k: v[r * round_size: (r + 1) * round_size]
                    for k, v in input.items()
                }

                # Generate!
                with torch.inference_mode():
                    try:
                        outputs = model.generate(
                            **truncated_input, max_new_tokens=512
                        )
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        if "out of memory" in str(e) and batch_size == 1:
                            raise Exception(
                                "GPU OOM --- Try using a larger GPU or more GPUs to user DIPPER"
                            ) from e
                        elif "out of memory" in str(e):
                            batch_size = batch_size // 2
                            error = True
                            break
                        else:
                            raise e

                text_output += (
                        " ".join(
                            tokenizer.batch_decode(
                                outputs, skip_special_tokens=True
                            )
                        )
                        + " "
                )

            if not error:
                torch.cuda.empty_cache()
                break

        return text_output
    except Exception:
        print("Dipper error")
        return text_output


### PROCESSES ###


def openai_process(openai_queue, api_key=None, cache=None):
    if api_key is not None:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    while True:
        task = openai_queue.get(block=True)
        if task is None:
            return

        (
            model,
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            prompt,
            system_prompt,
            max_tokens,
            timeout,
            logprobs,
            echo,
            destination_queue,
        ) = task
        destination_queue.put(
            call_openai(
                model,
                temperature,
                top_p,
                presence_penalty,
                frequency_penalty,
                prompt,
                system_prompt,
                max_tokens,
                timeout,
                logprobs,
                echo,
                client,
                cache,
            )
        )


def dipper_server(queue, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in devices)
    device = "cuda" if len(devices) else "cpu"
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    # Load model. Hopefully can fit on a single GPU. If not, this needs to be adapted to use CPU or multi-GPU using a memory map.
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    model = T5ForConditionalGeneration.from_pretrained(
        "kalpeshk2011/dipper-paraphraser-xxl", torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # Await tasks
    while True:
        task = queue.get(block=True)
        if task is None:
            return

        texts, destination_queue = task
        destination_queue.put(call_dipper(model, tokenizer, texts, device))


def custom_model_process(custom_model_queue, model_path, devices, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in devices)
    # device = "cuda" if len(devices) else "cpu"
    model_kwargs = get_server_args(config)
    model_kwargs["max_model_len"] = config.max_new_tokens * 2
    server = LLM(model_path,  **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    gen_params = SamplingParams(
        temperature=config.custom_temperature,
        max_tokens=config.max_new_tokens,
        n=config.custom_batch,
    )
    system_prompt = """
        You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences.\n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. Do not include any information that is not present in the original text. Do not respond with a greeting or any other extraneous information. Skip the preamble. Just rewrite the text directly.
    """
    instruction = "Paraphrase the following text:\n[[START OF TEXT]]\n{}\n[[END OF TEXT]]"
    response = "[[START OF PARAPHRASE]]\n"

    while True:
        task = custom_model_queue.get(block=True)
        text, destination_queue = task
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction.format(text)},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message = True
        )
        output = server.generate([prompt], gen_params, use_tqdm=False)[0]
        paraphrased = []
        for output_text in output.outputs:
            paraphrased.append(output_text.text.strip())
        destination_queue.put(paraphrased)
        torch.cuda.empty_cache()


def translate_process(translation_queue, langs, device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    import argostranslate.package
    import argostranslate.settings
    import argostranslate.translate

    if device != "cpu":
        os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
        argostranslate.settings.device = "cuda"
    else:
        os.environ["ARGOS_DEVICE_TYPE"] = "cpu"
        argostranslate.settings.device = "cpu"

    # Install translation models
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    def install_model(la, lb):
        package_to_install = next(
            filter(
                lambda x: (x.from_code == la and x.to_code == lb),
                available_packages,
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())

    for i, li in enumerate(langs):
        for j, lj in enumerate(langs):
            if i == j:
                continue
            try:
                install_model(li, lj)
            except Exception:
                pass

    # Get actual models
    pairs = {}

    while True:
        task = translation_queue.get(block=True)
        if task is None:
            return

        text, la, lb, dst_queue = task
        if (la, lb) not in pairs:
            pairs[
                (la, lb)
            ] = argostranslate.translate.get_translation_from_codes(la, lb)

        try:
            dst_queue.put(pairs[(la, lb)].translate(text))
        except RuntimeError as e:
            print(e)
            print("Reducing number of CUDA threads")
            translation_queue.put(task)
            return
