#!/bin/bash

tar -xvf run/static_data/encodings.tar.gz
tar -xvf run/static_data/selected.tar.gz
mv encodings/ run/static_data/
mv selected/ run/static_data/

git submodule init
git submodule update

export CUDA
conda install python=3.9 openssl ninja cmake
pip install --upgrade pip setuptools
pip install submodules/vllm
pip install lingua-language-detector tiktoken transformers scikit-learn nltk pyinflect accelerate openai textattack pandas dacite dahuffman argostranslate dill mauve-text faiss-gpu sacrebleu sacremoses psutil pyext xformers peft trl wandb deepspeed
pip install -e submodules/apps
python setup.py install
pip install --upgrade openai httpcore
cd submodules/cpp-hash && python setup.py install
cd ../..



