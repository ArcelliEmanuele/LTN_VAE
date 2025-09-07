# =============================================================================
# analyze.py
# -*- coding: utf-8 -*-
"""
Script 2: Analysis from Enhanced Checkpoints (v13.0) - STYLED v2

This script is responsible for the second phase of the workflow.
It MUST be run after `train.py` has completed.

1.  Loads the per-epoch enhanced model checkpoints (.pt files).
2.  Extracts both model weights and saved training metrics from each checkpoint.
3.  Performs in-depth, per-epoch analysis (Probing, VAE, Manifold).
4.  Combines the extracted training metrics with new analysis metrics into a final
    JSON file for each run.
5.  Performs final, summary analyses (Causal Intervention).
6.  Generates all plots and the final summary table with a consistent,
    publication-quality style, using the 'viridis' colormap.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ltn
import json
import glob
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Optional
import argparse
warnings.filterwarnings("ignore", category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid')
# --- Imports for Hardware Monitoring ---
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    warnings.warn("pynvml not found. GPU monitoring will be disabled.")

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    warnings.warn("fvcore not found. FLOPs calculation will be disabled.")

# =============================================================================
# 1. CONFIGURATION (IMPORTANT: Must match train.py)
# =============================================================================
N_RUNS = 3
N_EPOCHS_BASELINE = 10
N_EPOCHS_LOGIC = 10
BATCH_SIZE = 128

# --- Probing/VAE Configuration ---
N_PROBE_REPEATS = 2
N_EPOCHS_VAE_PER_PROBE = 21

# =============================================================================
# 2. SETUP: DEVICE, DIRECTORIES, AND HELPERS
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# 3. DATA HANDLING
# =============================================================================
def get_mnist_dataset_for_digits_addition() -> Tuple[list, list]:
    n_train_examples, n_test_examples, n_operands = 30000, 5000, 2
    mnist_train = torchvision.datasets.MNIST("./datasets/", train=True, download=True, transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST("./datasets/", train=False, download=True, transform=torchvision.transforms.ToTensor())
    train_imgs, train_labels = mnist_train.data / 255.0, mnist_train.targets
    imgs_operand_train = [train_imgs[i * n_train_examples:(i + 1) * n_train_examples] for i in range(n_operands)]
    labels_operand_train = [train_labels[i * n_train_examples:(i + 1) * n_train_examples] for i in range(n_operands)]
    test_imgs, test_labels = mnist_test.data / 255.0, mnist_test.targets
    imgs_operand_test = [test_imgs[i * n_test_examples:(i + 1) * n_test_examples] for i in range(n_operands)]
    labels_operand_test = [test_labels[i * n_test_examples:(i + 1) * n_test_examples] for i in range(n_operands)]
    train_set = [torch.stack(imgs_operand_train, dim=1).unsqueeze(2), labels_operand_train[0] + labels_operand_train[1], torch.stack(labels_operand_train, dim=1), torch.stack(imgs_operand_train, dim=1)]
    test_set = [torch.stack(imgs_operand_test, dim=1).unsqueeze(2), labels_operand_test[0] + labels_operand_test[1], torch.stack(labels_operand_test, dim=1), torch.stack(imgs_operand_test, dim=1)]
    return train_set, test_set

class DataLoader(object):
    def __init__(self, fold: list, batch_size: int = 1, shuffle: bool = True):
        self.fold, self.batch_size, self.shuffle = fold, batch_size, shuffle
        self.num_samples = self.fold[0].shape[0]
    def __len__(self) -> int:
        return int(np.ceil(self.num_samples / self.batch_size))
    def __iter__(self) -> Tuple[torch.Tensor, ...]:
        idxlist = list(range(self.num_samples))
        if self.shuffle: np.random.shuffle(idxlist)
        for _, start_idx in enumerate(range(0, self.num_samples, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            yield (self.fold[0][idxlist[start_idx:end_idx]], self.fold[1][idxlist[start_idx:end_idx]], self.fold[2][idxlist[start_idx:end_idx]], self.fold[3][idxlist[start_idx:end_idx]])

# =============================================================================
# 4. MODELS AND LOGIC
# =============================================================================
class MNISTFeatureExtractor(nn.Module):
    def __init__(self):
        super(MNISTFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 100)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.tanh = nn.Tanh()
        self.batch_norm = nn.BatchNorm1d(100)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        activations = {}
        out = self.maxpool(self.relu(self.conv1(x))); activations['conv1'] = out
        out = self.maxpool(self.relu(self.conv2(out))); activations['conv2'] = out
        out = torch.flatten(out, 1)
        out = self.fc1(out); activations['fc1_pre_act'] = out
        out = self.tanh(self.batch_norm(out)); activations['features'] = out
        return activations

class SingleDigitClassifier(nn.Module):
    def __init__(self):
        super(SingleDigitClassifier, self).__init__()
        self.feature_extractor = MNISTFeatureExtractor()
        self.classifier_head = nn.Sequential(
            nn.Linear(100, 84), nn.BatchNorm1d(84), nn.Tanh(), nn.Linear(84, 10))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = self.feature_extractor(x)
        features = activations['features']
        return self.classifier_head(features)

class LogitsToPredicate(nn.Module):
    def __init__(self, logits_model: nn.Module):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        logits = self.logits_model(x.to(DEVICE))
        probs = self.softmax(logits)
        return torch.gather(probs, 1, d.long())


# =============================================================================
# 5. ANALYSIS FUNCTIONS (PROBING, VAE, CAUSAL, PLOTTING)
# =============================================================================

# --- Unified Plotting Style Helper ---
def apply_plot_styles(
    ax: plt.Axes,
    title: str,
    xlabel: str = '',
    ylabel: str = '',
    cbar: Optional[plt.colorbar] = None
):
    """
    Applies a consistent, publication-quality style to a Matplotlib axis.
    """
    # Set titles and labels with specified font sizes and color
    ax.set_title(title, fontsize=22, color='black', pad=20)
    ax.set_xlabel(xlabel, fontsize=18, color='black', labelpad=15)
    ax.set_ylabel(ylabel, fontsize=18, color='black', labelpad=15)

    # Style the ticks and spines (the plot border)
    ax.tick_params(axis='x', colors='black', labelsize=14)
    ax.tick_params(axis='y', colors='black', labelsize=14)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    # Style the legend if it exists
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color('black')
            text.set_fontsize(14)
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')

    # Style the colorbar if it exists
    if cbar:
        cbar.set_label(ylabel, rotation=270, labelpad=25, color='black', fontsize=18)
        cbar.ax.yaxis.set_tick_params(color='black')
        cbar.ax.tick_params(axis='y', labelcolor='black', labelsize=14)

    # Apply a light grid for better readability
    ax.grid(True, color='lightgray', linestyle='--')


# --- Probing & VAE Helpers ---
def extract_hidden_states_and_images(model, data_loader, is_ltn, layer_name):
    model.eval()
    if is_ltn:
        feature_extractor = model.model.logits_model.feature_extractor
    else:
        feature_extractor = model.feature_extractor

    all_states, all_labels, all_raw_images = [], [], []
    with torch.no_grad():
        for operand_images, _, digit_labels, raw_images in data_loader:
            for i in range(2): # For each of the two operands
                img_tensor, labels, raw_img_tensor = operand_images[:, i].to(DEVICE), digit_labels[:, i].cpu().numpy(), raw_images[:, i]
                states = feature_extractor(img_tensor)[layer_name].cpu().numpy()
                if states.ndim > 2: states = states.reshape(states.shape[0], -1)
                all_states.append(states)
                all_labels.append(labels)
                all_raw_images.append(raw_img_tensor.cpu().numpy())

    return np.concatenate(all_states), np.concatenate(all_labels), np.concatenate(all_raw_images)

def extract_hidden_states(model, data_loader, is_ltn, layer_name):
    states, labels, _ = extract_hidden_states_and_images(model, data_loader, is_ltn, layer_name)
    return states, labels

# --- Probing ---
def run_probes_for_epoch(model, probe_loader, is_ltn, layers_to_probe: List[str]):
    probe_accuracies = {}
    for layer in layers_to_probe:
        states, labels = extract_hidden_states(model, probe_loader, is_ltn, layer)
        accuracies = []
        for i in range(N_PROBE_REPEATS):
            X_train, X_test, y_train, y_test = train_test_split(states, labels, test_size=0.3, random_state=42 + i, stratify=labels)
            probe = LogisticRegression(max_iter=1000, random_state=42 + i, solver='lbfgs', n_jobs=-1).fit(X_train, y_train)
            accuracies.append(accuracy_score(y_test, probe.predict(X_test)))
        probe_accuracies[layer] = np.mean(accuracies)
    return probe_accuracies

# --- VAE Analysis ---
class VAEProbe(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64, latent_dim=2):
        super(VAEProbe, self).__init__(); self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim * 2)); self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim), nn.Tanh())
    def reparameterize(self, mu, log_var): std = torch.exp(0.5 * log_var); eps = torch.randn_like(std); return mu + eps * std
    def forward(self, x): h = self.encoder(x); mu, log_var = torch.chunk(h, 2, dim=-1); z = self.reparameterize(mu, log_var); return self.decoder(z), mu, log_var

def vae_loss_function(recon_x, x, mu, log_var): MSE = F.mse_loss(recon_x, x, reduction='sum'); KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()); return MSE + KLD

def run_per_epoch_vae_analysis(model, data_loader, is_ltn, model_name, run_num, epoch):
    print(f"    -> Running VAE Analysis for Epoch {epoch} on Run {run_num+1}..."); all_states, all_labels = extract_hidden_states(model, data_loader, is_ltn, 'features')
    vae = VAEProbe().to(DEVICE); optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(all_states).float()); loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True); vae.train()
    for _ in range(N_EPOCHS_VAE_PER_PROBE):
        for data_batch in loader:
            data = data_batch[0].to(DEVICE); optimizer.zero_grad(); recon_batch, mu, log_var = vae(data)
            loss = vae_loss_function(recon_batch, data, mu, log_var); loss.backward(); optimizer.step()

    vae.eval()
    with torch.no_grad(): h = vae.encoder(torch.from_numpy(all_states).float().to(DEVICE)); mu, _ = torch.chunk(h, 2, dim=-1); z = mu.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(z[:, 0], z[:, 1], c=all_labels, cmap='inferno', alpha=0.7, s=15)
    cbar = fig.colorbar(scatter, ticks=np.arange(10), ax=ax)

    # Use the centralized styling function
    apply_plot_styles(
        ax=ax,
        title=f"Latent Space: {model_name} | Run {run_num+1}, Epoch {epoch}",
        xlabel="Latent Dimension 1",
        ylabel="Digit", # Y-label is used for the colorbar here
        cbar=cbar
    )

    plt.tight_layout()
    plt.savefig(f"latent_space_scatter_plots/{model_name.replace(' ', '_')}_run{run_num}_epoch{epoch}.png", facecolor='white'); plt.close(fig)

# --- Generative Manifold Analysis ---
def run_generative_manifold_analysis(model, data_loader, is_ltn, model_name, run_num, epoch):
    print(f"    -> Plotting Generative Manifold for Epoch {epoch} on Run {run_num+1}...")
    all_states, _, all_images = extract_hidden_states_and_images(model, data_loader, is_ltn, 'features')

    vae = VAEProbe().to(DEVICE); optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(all_states).float()); loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True); vae.train()
    for _ in range(N_EPOCHS_VAE_PER_PROBE):
        for data_batch in loader:
            data = data_batch[0].to(DEVICE); optimizer.zero_grad(); recon_batch, mu, log_var = vae(data)
            loss = vae_loss_function(recon_batch, data, mu, log_var); loss.backward(); optimizer.step()

    vae.eval()
    with torch.no_grad(): h = vae.encoder(torch.from_numpy(all_states).float().to(DEVICE)); mu, _ = torch.chunk(h, 2, dim=-1); z_coords = mu.cpu().numpy()

    n_images_per_dim = 20; grid_x = np.linspace(z_coords[:, 0].min(), z_coords[:, 0].max(), n_images_per_dim); grid_y = np.linspace(z_coords[:, 1].max(), z_coords[:, 1].min(), n_images_per_dim) # Flipped for imshow
    img_height, img_width = all_images[0].shape; manifold = np.zeros((img_height * n_images_per_dim, img_width * n_images_per_dim))

    grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
    distances = cdist(grid_points, z_coords); closest_img_indices = np.argmin(distances, axis=1)

    for i, img_idx in enumerate(closest_img_indices):
        row, col = i // n_images_per_dim, i % n_images_per_dim
        manifold[row * img_height : (row + 1) * img_height, col * img_width : (col + 1) * img_width] = all_images[img_idx]

    fig, ax = plt.subplots(figsize=(15, 15)); ax.imshow(manifold, cmap='gray')
    ax.set_xticks([]); ax.set_yticks([])

    # Use the centralized styling function
    apply_plot_styles(
        ax=ax,
        title=f"Generative Manifold: {model_name} | Run {run_num+1}, Epoch {epoch}"
    )

    plt.tight_layout()
    os.makedirs("generative_manifold_plots", exist_ok=True)
    plt.savefig(f"generative_manifold_plots/{model_name.replace(' ', '_')}_run{run_num}_epoch{epoch}.png", facecolor='white', bbox_inches='tight')
    plt.close(fig)


# --- Causal Intervention ---
def find_all_misclassified_examples(model, data_loader):
    misclassified_examples = []; model.eval()
    with torch.no_grad():
        for operand_images, _, digit_labels, _ in data_loader:
            img, true_digit = operand_images[:, 0].to(DEVICE), digit_labels[:, 0].cpu().numpy()
            pred_digit = torch.argmax(model(img), dim=1).cpu().numpy()
            for i in range(img.shape[0]):
                if pred_digit[i] != true_digit[i]: misclassified_examples.append((img[i].unsqueeze(0), true_digit[i]))
    return misclassified_examples

def run_causal_intervention_for_epoch(bad_model, good_model, data_loader):
    good_feature_extractor = good_model.logits_model.feature_extractor.eval()
    misclassified_examples = find_all_misclassified_examples(bad_model, data_loader)
    if not misclassified_examples: return 100.0

    success_count = 0
    for misclassified_img, true_digit in misclassified_examples:
        with torch.no_grad(): good_activation = good_feature_extractor(misclassified_img)['fc1_pre_act']
        def patch_activation(module, input, output): return good_activation
        handle = bad_model.feature_extractor.fc1.register_forward_hook(patch_activation)
        with torch.no_grad(): patched_prediction = torch.argmax(bad_model(misclassified_img)).item()
        handle.remove()
        if patched_prediction == true_digit: success_count += 1

    return (success_count / len(misclassified_examples)) * 100

# --- Metric Aggregation and Plotting ---
def load_and_aggregate_metrics(model_name: str, metrics_dir: str = "run_metrics"):
    metric_files = glob.glob(os.path.join(metrics_dir, f"{model_name.replace(' ', '_')}_run*.json"))
    if not metric_files: print(f"Warning: No metric files found for model '{model_name}'."); return None
    all_runs_data = [json.load(open(f_path, 'r')) for f_path in metric_files]
    aggregated = {}
    if not all_runs_data or not all_runs_data[0]: return None
    for key in all_runs_data[0].keys():
        if key == 'probe_accuracies':
            if all('probe_accuracies' in run and run['probe_accuracies'] and run['probe_accuracies'][0] for run in all_runs_data):
                layers = all_runs_data[0]['probe_accuracies'][0].keys(); aggr_probes = {}
                for layer in layers:
                    layer_data_matrix = [[epoch_probes[layer] for epoch_probes in run['probe_accuracies']] for run in all_runs_data]
                    aggr_probes[layer] = {'mean': np.mean(layer_data_matrix, axis=0), 'std': np.std(layer_data_matrix, axis=0, ddof=1)}
                aggregated['probe_accuracies'] = aggr_probes
        elif isinstance(all_runs_data[0].get(key), list):
             data_matrix = [run[key] for run in all_runs_data if key in run and len(run[key]) == len(all_runs_data[0][key])]
             if len(data_matrix) == len(all_runs_data):
                 valid_data = [np.array(d, dtype=float) for d in data_matrix]
                 aggregated[key] = {'mean': np.nanmean(valid_data, axis=0), 'std': np.nanstd(valid_data, axis=0, ddof=1)}
    return aggregated

def plot_all_metrics(model_names: List[str], n_epochs_map: Dict[str, int]):
    print("\n--- Generating Plots ---")
    all_res = {name: load_and_aggregate_metrics(name) for name in model_names}
    os.makedirs("metric_plots", exist_ok=True)
    
    # --- Generate a beautiful color scheme from the 'viridis' colormap ---
    colormap = cm.get_cmap('viridis')
    # We sample from 0 to 0.9 to avoid the bright yellow at the end, which is hard to see.
    color_samples = [colormap(i) for i in np.linspace(0, 0.4, len(model_names))]
    colors = {name: color for name, color in zip(model_names, color_samples)}


    # --- Plot 1: Task Performance ---
    fig, axs = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)
    metric_map = [('test_accuracy_sum', 'Test Sum Accuracy'), ('test_accuracy_digit', 'Test Digit Accuracy')]
    for ax, (m_key, title) in zip(axs, metric_map):
        for name in model_names:
            res = all_res.get(name)
            if res and m_key in res:
                mean, std = res[m_key]['mean'], res[m_key]['std']; epochs = range(1, len(mean) + 1)
                ax.plot(epochs, mean, color=colors.get(name, 'k'), label=name, linewidth=2.5)
                ax.fill_between(epochs, mean - std, mean + std, color=colors.get(name, 'k'), alpha=0.2)
        ax.legend(); ax.set_ylim(0, 1.05)
        apply_plot_styles(ax, title=title, xlabel='Epochs', ylabel='Accuracy')
    plt.savefig("metric_plots/final_task_performance_comparison.png", facecolor='white', bbox_inches='tight'); plt.close(fig)

    # --- Plot 2: Probing Performance ---
    if any(res and 'probe_accuracies' in res for res in all_res.values()):
        probe_res = all_res['Baseline']['probe_accuracies'] if 'Baseline' in all_res and 'probe_accuracies' in all_res['Baseline'] else None
        if probe_res:
            layers = probe_res.keys()
            fig, axs = plt.subplots(1, len(layers), figsize=(len(layers) * 8, 7), constrained_layout=True)
            if len(layers) == 1: axs = [axs]
            for ax, layer in zip(axs, layers):
                for name in model_names:
                    res = all_res.get(name)
                    if res and 'probe_accuracies' in res and layer in res['probe_accuracies']:
                        mean, std = res['probe_accuracies'][layer]['mean'], res['probe_accuracies'][layer]['std']; epochs = range(1, len(mean) + 1)
                        ax.plot(epochs, mean, color=colors.get(name, 'k'), label=name, linewidth=2.5)
                        ax.fill_between(epochs, mean - std, mean + std, color=colors.get(name, 'k'), alpha=0.2)
                ax.legend()
                apply_plot_styles(ax, title=f"Probe on '{layer}'", xlabel='Epochs', ylabel='Probe Accuracy')
            plt.savefig("metric_plots/final_layerwise_probe_comparison.png", facecolor='white', bbox_inches='tight'); plt.close(fig)

    # --- Plot 3: Causal Intervention ---
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    causal_models_found = False
    for name in ['Baseline']: # This plot currently only supports one model
        res = all_res.get(name)
        if res and 'causal_success_rate' in res and res['causal_success_rate'].get('mean') is not None and len(res['causal_success_rate']['mean']) > 0:
            mean, std = res['causal_success_rate']['mean'], res['causal_success_rate']['std']
            epochs = range(1, len(mean) + 1)
            # Use the single model's color from our new colormap
            ax.plot(epochs, mean, color=colors.get(name, 'k'), label=f"{name} Intervention Success", linewidth=2.5)
            ax.fill_between(epochs, mean - std, mean + std, color=colors.get(name, 'k'), alpha=0.2)
            causal_models_found = True

    if causal_models_found:
        ax.legend(); ax.set_ylim(0, 101)
        apply_plot_styles(ax, title='Causal Intervention Success Rate vs. Epoch', xlabel='Epoch', ylabel='Success Rate (%)')
        plt.savefig("metric_plots/causal_intervention_comparison.png", facecolor='white', bbox_inches='tight')
        print("Saved Causal Intervention plot to 'metric_plots/causal_intervention_comparison.png'")
    else:
        print("Skipping causal plot: No analysis data found.")
    plt.close(fig)

    print("All plots saved to 'metric_plots/'.")


# =============================================================================
# 6. HIGH-LEVEL WORKFLOW FUNCTIONS
# =============================================================================
def calculate_and_save_all_metrics(models_to_run: Dict, args: argparse.Namespace):
    """
    Loads checkpoints, runs all per-epoch analyses (probe, VAE, manifold, causal),
    and saves the consolidated metrics to a JSON file for each run.
    """
    print("\n" + "="*50 + "\nPHASE 1: PER-EPOCH ANALYSIS & METRIC GENERATION\n" + "="*50)
    # Load data only when it's needed for calculations
    _, test_set = get_mnist_dataset_for_digits_addition()
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    for name, config in models_to_run.items():
        print(f"\n--- Analyzing Model: {name} ---")
        is_ltn = config["type"] == "logic"

        for i in range(N_RUNS):
            print(f"  > Processing Run {i+1}/{N_RUNS}")
            metric_fname = f"run_metrics/{name.replace(' ', '_')}_run{i}_metrics.json"
            if os.path.exists(metric_fname):
                with open(metric_fname, 'r') as f:
                    final_run_metrics = json.load(f)
            else:
                final_run_metrics = {}

            # If analysis flags are passed, clear any previously saved results for
            # those analyses to force fresh recalculation.
            if args.probe or args.probe_final_layer:
                print("  > Probe flag detected, clearing previous probe results for this run.")
                final_run_metrics["probe_accuracies"] = []
            if args.causal:
                print("  > Causal flag detected, clearing previous causal results for this run.")
                final_run_metrics["causal_success_rate"] = []

            # Ensure lists exist even if file was empty
            final_run_metrics.setdefault("probe_accuracies", [])
            final_run_metrics.setdefault("causal_success_rate", [])


            for epoch in tqdm(range(1, config["epochs"] + 1), desc="  Epoch Analysis"):
                # Always load the base metrics from the checkpoint first
                model = SingleDigitClassifier().to(DEVICE)
                if is_ltn: model = ltn.Predicate(LogitsToPredicate(model))
                checkpoint_path = f"checkpoints/epochs/{name.replace(' ', '_')}_run{i}_epoch{epoch}.pt"
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                except FileNotFoundError as e:
                    print(f"ERROR: {e}. Run train.py first.")
                    exit()
                except RuntimeError:
                    print(f"INFO: Could not load full checkpoint object for {name} Run {i} Epoch {epoch}. Loading model weights only.")
                    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
                    checkpoint = {'model_state_dict': model.state_dict()}

                model.load_state_dict(checkpoint['model_state_dict'])

                # Load metrics saved during training (e.g., accuracy, loss)
                for key in checkpoint.keys():
                    if key not in ['model_state_dict', 'optimizer_state_dict']:
                        if key not in final_run_metrics or len(final_run_metrics.get(key, [])) < epoch:
                             final_run_metrics.setdefault(key, []).append(checkpoint[key])


                # --- Run Analyses That Are Calculated in this Script ---
                if args.probe or args.probe_final_layer:
                    layers = ['features'] if args.probe_final_layer else ['conv1', 'conv2', 'fc1_pre_act', 'features']
                    final_run_metrics["probe_accuracies"].append(run_probes_for_epoch(model, test_loader, is_ltn, layers))

                if args.causal and name in ['Baseline', 'Fuzzy LTN']:
                    good_model_ltn = ltn.Predicate(LogitsToPredicate(SingleDigitClassifier().to(DEVICE)))
                    good_model_chpt_path = f"checkpoints/epochs/Stable_LTN_run{i}_epoch{epoch}.pt"
                    if os.path.exists(good_model_chpt_path):
                        good_model_chpt = torch.load(good_model_chpt_path, map_location=DEVICE, weights_only=False)
                        good_model_ltn.load_state_dict(good_model_chpt['model_state_dict'])
                        bad_model_classifier = model.model.logits_model if is_ltn else model
                        good_model_wrapper = good_model_ltn.model
                        final_run_metrics["causal_success_rate"].append(run_causal_intervention_for_epoch(bad_model_classifier, good_model_wrapper, test_loader))
                    else:
                        print(f"  [WARN] Causal analysis skipped for {name} epoch {epoch}: Stable_LTN checkpoint not found.")
                        final_run_metrics["causal_success_rate"].append(None)
                
                # VAE and Manifold plots are always generated if the flag is present
                # as they are visual diagnostics not stored in the metrics file.
                if args.vae and i == args.vae_run:
                    run_per_epoch_vae_analysis(model, test_loader, is_ltn, name, i, epoch)
                if args.manifold and i == args.vae_run:
                    run_generative_manifold_analysis(model, test_loader, is_ltn, name, i, epoch)


            # Clean up metric lists to match number of epochs and save
            for key, val in final_run_metrics.items():
                if key not in ['model_state_dict', 'optimizer_state_dict']:
                    final_run_metrics[key] = val[:config['epochs']]

            with open(metric_fname, 'w') as f:
                json.dump(final_run_metrics, f, indent=4)
            print(f"  > Saved/Updated consolidated metrics for Run {i+1} to {metric_fname}")

def generate_final_plots_and_summary(models_to_run: Dict, args: argparse.Namespace):
    """
    Loads saved metrics from JSON files to generate all summary plots and the
    final results table. This function does not perform any new analysis.
    """
    if args.plot:
        print("\n" + "="*50 + "\nPHASE 2: PLOTTING FROM SAVED METRICS\n" + "="*50)
        plot_all_metrics(list(models_to_run.keys()), {name: config['epochs'] for name, config in models_to_run.items()})

    if args.summary:
        print("\n" + "="*50 + "\nFINAL SUMMARY TABLE\n" + "="*50)
        summary_data = { "Model": [], "Final Sum Accuracy": [], "Final Probe (features)": [], "Final Causal Success (%)": []}
        for name in models_to_run.keys():
            aggregated = load_and_aggregate_metrics(name)
            if not aggregated: continue
            summary_data["Model"].append(name)
            summary_data["Final Sum Accuracy"].append(f"{aggregated.get('test_accuracy_sum',{}).get('mean', [np.nan])[-1]:.4f}")
            probe_acc = aggregated.get('probe_accuracies', {}).get('features', {})
            summary_data["Final Probe (features)"].append(f"{probe_acc.get('mean', [np.nan])[-1]:.4f}" if probe_acc and probe_acc.get('mean') is not None else "N/A")
            causal_rate = aggregated.get('causal_success_rate', {})
            summary_data["Final Causal Success (%)"].append(f"{causal_rate.get('mean', [np.nan])[-1]:.2f}" if (name in ['Baseline', 'Fuzzy LTN'] and causal_rate and np.any(causal_rate.get('mean'))) else "N/A")

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('summary.csv', index=False)
        if not summary_df.empty:
            print(summary_df.to_markdown(index=False))
        else:
            print("Summary DataFrame could not be generated. Run analysis first.")

# =============================================================================
# 7. MAIN ANALYSIS PIPELINE
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run analysis on trained models. Run train.py first.")
    parser.add_argument('--all', action='store_true', help='Run all analysis steps (full probe, vae, manifold, causal, plot, summary).')
    parser.add_argument('--probe', action='store_true', help='(Re)calculate linear probing on all specified layers.')
    parser.add_argument('--probe-final-layer', action='store_true', help='(Re)calculate linear probing on the final layer only (fast).')
    parser.add_argument('--vae', action='store_true', help='Run per-epoch VAE analysis.')
    parser.add_argument('--manifold', action='store_true', help='Run per-epoch generative manifold analysis.')
    parser.add_argument('--vae-run', type=int, default=N_RUNS-1, metavar='R', help=f'0-based index of the run for VAE/manifold analysis (default: {N_RUNS-1}).')
    parser.add_argument('--causal', action='store_true', help='(Re)calculate per-epoch causal intervention analysis.')
    parser.add_argument('--plot', action='store_true', help='Generate and save all plots from existing metric files.')
    parser.add_argument('--summary', action='store_true', help='Print the final summary table from existing metric files.')
    args = parser.parse_args()

    if args.all:
        args.probe = args.vae = args.manifold = args.causal = args.plot = args.summary = True
        # Use a sensible default for probe_final_layer when --all is not specified
        if not args.probe: args.probe_final_layer = True


    if args.vae and not (0 <= args.vae_run < N_RUNS):
        parser.error(f"--vae-run must be an index between 0 and {N_RUNS-1}.")

    # Define which analysis tasks are for calculation vs. plotting/reporting
    calculation_tasks = [args.probe, args.probe_final_layer, args.vae, args.causal, args.manifold]
    reporting_tasks = [args.plot, args.summary]

    if not any(calculation_tasks) and not any(reporting_tasks):
        parser.print_help()
        print("\nNo task specified. Use --all or specific flags like --probe, --plot, etc. Exiting.")
        exit()

    # Shared setup
    set_seed(42)
    models_to_run = {"Baseline": {"type": "simple", "epochs": N_EPOCHS_BASELINE}, "Fuzzy LTN": {"type": "logic", "epochs": N_EPOCHS_LOGIC}, "Stable LTN": {"type": "logic", "epochs": N_EPOCHS_LOGIC}}
    os.makedirs("latent_space_scatter_plots", exist_ok=True)
    os.makedirs("generative_manifold_plots", exist_ok=True)
    os.makedirs("run_metrics", exist_ok=True)
    os.makedirs("checkpoints/epochs", exist_ok=True)

    # --- Phase 1: Calculate and Save Metrics ---
    if any(calculation_tasks):
        calculate_and_save_all_metrics(models_to_run, args)

    # --- Phase 2: Generate Plots and Summary from Saved Metrics ---
    if any(reporting_tasks):
        generate_final_plots_and_summary(models_to_run, args)

    print("\n" + "="*50 + "\nANALYSIS SCRIPT COMPLETE\n" + "="*50)