import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
# pdf for plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

from watermark_benchmark import Generation


def mean_confidence_interval(data, confidence=0.95):
    # Convert data to numpy array
    data = np.array(data)

    # Calculate mean
    mean = np.mean(data)

    # Standard error of the mean (SEM)
    sem = stats.sem(data)

    # Z-score for the given confidence level (95% confidence -> Z = 1.96)
    z_score = stats.norm.ppf((1 + confidence) / 2.0)

    # Calculate margin of error using Z-distribution
    margin_of_error = sem * z_score


    return mean, margin_of_error

def plot_needed_paraphrases(input_file: str, output_dir: str):
    generations = Generation.from_file(input_file)

    # Group generations by watermark and id
    grouped = {}
    for gen in generations:
        if gen.watermark is None:
            continue
        if gen.watermark.generator not in grouped:
            grouped[gen.watermark.generator] = defaultdict(list)
        grouped[gen.watermark.generator][gen.id].append(gen)
    all_data = {}
    for watermark in grouped:
        watermark_data = []
        watermark_group = grouped[watermark]
        for n_samples in range(1, 17):
            id_data = []
            for id in tqdm(watermark_group):
                samples = [g for g in watermark_group[id] if g.attack is not None]
                sample = random.sample(samples, n_samples)
                pvals = [g.pvalue for g in sample if g.rating > 0.8]
                max_pval = max(pvals) if pvals else 0
                value = 0 if max_pval < 0.1 else 1
                id_data.append(value)
            mean, ci = mean_confidence_interval(id_data)
            watermark_data.append((n_samples, mean, ci))
        all_data[watermark] = watermark_data

    # Plot the data
    fig, ax = plt.subplots()
    for watermark in all_data:
        data = all_data[watermark]
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        ci = [d[2] for d in data]
        #fill between
        ax.plot(x, y, label=watermark)
        ax.fill_between(x, np.array(y) - np.array(ci), np.array(y) + np.array(ci), alpha=0.2)
    ax.set_xlabel('Number of Paraphrases')
    ax.set_ylabel('Evasion Rate ↓')
    ax.title.set_text('Non-Adaptive Paraphrasing With PValue = 0.1')
    ax.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/needed_paraphrases.pdf")



# Usage
input_file = "run_datagen/results/detect.tsv"  # Replace with your input file name
output_dir = "plots"  # Replace with your desired output directory
plot_needed_paraphrases(input_file, output_dir)
