# -*- coding: utf-8 -*-
"""
Script 5: Analysis to LaTeX Table using Pandas

This script performs the full analysis and formats the final report
as a LaTeX table using the pandas library.
"""
import os
import glob
import torch
import torch.nn as nn
import numpy as np
import ltn
from collections import defaultdict
import pandas as pd  # <-- Import pandas
from config import *

# =============================================================================
# 1. CRITICAL: MODEL DEFINITIONS
# =============================================================================
# IMPORTANT: You MUST copy your actual model class definitions from your
# `config.py` or other files here. The classes below are placeholders.
# The accuracy of the "Num. Parameters" column depends entirely on this section.

DEVICE = torch.device("cpu") # We only need CPU for parameter counting


# =============================================================================
# 2. ANALYSIS LOGIC (Unchanged)
# =============================================================================
CHECKPOINT_DIR = "checkpoints/epochs"

def get_model_parameter_counts():
    """Instantiates each model type to calculate the number of trainable parameters."""
    print("--- Calculating Model Parameters ---")
    param_counts = {}
    baseline_model = SingleDigitClassifier().to(DEVICE)
    param_counts["Baseline"] = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    core_ltn_model = SingleDigitClassifier().to(DEVICE)
    ltn_model = ltn.Predicate(LogitsToPredicate(core_ltn_model))
    ltn_params = sum(p.numel() for p in ltn_model.parameters() if p.requires_grad)
    param_counts["Fuzzy LTN"] = ltn_params
    param_counts["Stable LTN"] = ltn_params
    print("Parameter calculation complete.")
    return param_counts

def analyze_checkpoints_to_latex():
    """
    Loads checkpoints, aggregates metrics, and prints a LaTeX table.
    """
    param_counts = get_model_parameter_counts()

    print("\n--- Analyzing Resource Usage from Checkpoints ---")
    if not os.path.isdir(CHECKPOINT_DIR):
        print(f"Error: Checkpoint directory not found at '{CHECKPOINT_DIR}'")
        return
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, '*.pt'))
    if not checkpoint_files:
        print(f"Error: No checkpoint files (.pt) found in '{CHECKPOINT_DIR}'.")
        return

    model_data = defaultdict(lambda: {"times": [], "powers": [], "memories": []})
    print(f"Found {len(checkpoint_files)} checkpoint files to analyze...")
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        try:
            model_name_raw = filename.split('_run')[0]
            model_name = model_name_raw.replace('_', ' ')
        except IndexError:
            continue
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        model_data[model_name]["times"].append(checkpoint.get('epoch_time', 0))
        model_data[model_name]["powers"].append(checkpoint.get('gpu_power_watt', 0))
        model_data[model_name]["memories"].append(checkpoint.get('gpu_memory_mib', 0))

    # =============================================================================
    # 3. BUILD DATAFRAME AND EXPORT TO LATEX
    # =============================================================================
    print("\n" + "="*70 + "\nGenerating LaTeX Output\n" + "="*70)

    results_data = []
    sorted_model_names = sorted(model_data.keys())

    for model_name in sorted_model_names:
        data = model_data[model_name]
        
        # Get parameter count and format it as a string with commas
        num_params = param_counts.get(model_name, 0)
        params_str = f"{num_params:,}" if num_params > 0 else "N/A"

        # Calculate averages for other metrics
        avg_time = np.mean(data["times"]) if data["times"] else 0
        avg_power = np.mean(data["powers"]) if data["powers"] else 0
        avg_memory = np.mean(data["memories"]) if data["memories"] else 0
        
        # Append all data for one row to the list
        results_data.append({
            "Model": model_name,
            "Num. Parameters": params_str,
            "Avg. Epoch Time (s)": avg_time,
            "Avg. GPU Power (W)": avg_power if avg_power > 0 else "N/A",
            "Avg. GPU Memory (MiB)": avg_memory if avg_memory > 0 else "N/A"
        })

    # Create a pandas DataFrame from the list of results
    df = pd.DataFrame(results_data)

    # Define the desired column order
    column_order = [
        "Model", "Num. Parameters", "Avg. Epoch Time (s)", 
        "Avg. GPU Power (W)", "Avg. GPU Memory (MiB)"
    ]
    df = df[column_order]

    # Generate the LaTeX string from the DataFrame
    # - index=False removes the DataFrame's integer index from the output.
    # - caption and label are standard LaTeX table elements.
    # - float_format specifies formatting for floating point numbers.
    latex_output = df.to_latex(
        index=False,
        caption="Full Model Comparison of Performance and Resources.",
        label="tab:model_comparison",
        float_format="%.2f"
    )
    
    # Print the final LaTeX code
    print(latex_output)


if __name__ == '__main__':
    analyze_checkpoints_to_latex()