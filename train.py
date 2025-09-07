# train.py
# -*- coding: utf-8 -*-
"""
Script 1: Model Training and Enhanced Checkpointing (v13.0)

This script is responsible for the first phase of the workflow:
1.  Defining the model architectures and data loaders.
2.  Training all specified models (Baseline, Fuzzy LTN, Stable LTN) for N_RUNS.
3.  Saving an enhanced checkpoint (.pt) after EVERY epoch of EVERY run.
    - Each checkpoint is a dictionary containing the model's state_dict
      AND all relevant training/testing metrics for that epoch.

After running this script successfully, proceed to `analyze.py`.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import ltn
import glob
import time
from tqdm import tqdm
import warnings
from config import *

# --- Imports for Hardware Monitoring ---
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

# =============================================================================
# 1. CONFIGURATION (IMPORTANT: Must match analyze.py)
# =============================================================================

# =============================================================================
# 2. SETUP: DEVICE, DIRECTORIES, AND HELPERS
# =============================================================================
print(f"Using device: {DEVICE}")

OUTPUT_DIRS = [
    "checkpoints", "checkpoints/epochs", "metric_plots", "generative_manifolds",
    "latent_space_scatter_plots", "generalization_matrices", "run_metrics"
]

GPU_HANDLE = None
if DEVICE.type == 'cuda' and HAS_PYNVML:
    try:
        pynvml.nvmlInit()
        GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        print("pynvml initialized successfully for GPU monitoring.")
    except Exception as e:
        print(f"Could not initialize pynvml: {e}")
        GPU_HANDLE = None
        HAS_PYNVML = False

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_output_directories():
    """Removes files from previous runs to ensure a clean slate."""
    print("--- Cleaning up previous run outputs ---")
    for directory in OUTPUT_DIRS:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            files = glob.glob(os.path.join(directory, '*'))
            if not files: continue
            print(f"Cleaning directory: {directory}")
            for f in files:
                try:
                    if os.path.isdir(f):
                        nested_files = glob.glob(os.path.join(f, '*'))
                        for nf in nested_files: os.remove(nf)
                    else:
                        os.remove(f)
                except OSError as e:
                    print(f"  Error removing file {f}: {e}")


# =============================================================================
# 4. MODELS AND LOGIC
# =============================================================================

class Lukasiewicz_AND(ltn.fuzzy_ops.ConnectiveOperator):
    def __call__(self, x, y): return torch.clamp(x + y - 1, min=0)
class AggregMean(ltn.fuzzy_ops.AggregationOperator):
    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        if mask is not None:
            numerator = torch.sum(torch.where(~mask, torch.zeros_like(xs), xs)); denominator = torch.sum(mask)
            return torch.div(numerator, denominator) if denominator > 0 else torch.tensor(0.0, device=xs.device)
        return torch.mean(xs, dim=dim, keepdim=keepdim)
class AggregMeanError(ltn.fuzzy_ops.AggregationOperator):
    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        if mask is not None: xs = torch.where(~mask, 1., xs.double())
        return torch.amin(xs, dim=dim, keepdim=keepdim)
eps = 1e-4
def pi_0(x): return (1 - eps) * x + eps
def pi_1(x): return (1 - 2 * eps) * x + eps
class Stable_AND(ltn.fuzzy_ops.ConnectiveOperator):
    def __call__(self, x, y): return pi_1(pi_0(x) * pi_0(y))

# =============================================================================
# 5. MAIN TRAINING PIPELINE
# =============================================================================
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    #clean_output_directories()
    set_seed(42)
    train_set, test_set = get_mnist_dataset_for_digits_addition()

    # --- MODIFICATION START: Helper function to count parameters ---
    def count_parameters(model: torch.nn.Module) -> int:
        """Counts the number of trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # --- MODIFICATION END ---

    models_to_run = {
        "Baseline": {"type": "simple", "epochs": N_EPOCHS_BASELINE},
        "Fuzzy LTN": {"type": "logic", "epochs": N_EPOCHS_LOGIC, "params": {"And": ltn.Connective(Lukasiewicz_AND()), "Exists": ltn.Quantifier(AggregMean(), "e"), "Forall": ltn.Quantifier(AggregMeanError(), "f")}},
        "Stable LTN": {"type": "logic", "epochs": N_EPOCHS_LOGIC, "params": {"And": ltn.Connective(Stable_AND()), "Exists": ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), "e"), "Forall": ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), "f")}}
    }

    print("\n" + "="*50 + "\nPHASE 1: TRAINING & ENHANCED CHECKPOINTING\n" + "="*50)

    for name, config in models_to_run.items():
        print(f"\n--- Starting Training Phase for {name} ---")
        for i in range(N_RUNS):
            print(f"\n--- RUN {i+1}/{N_RUNS} for {name} ---"); set_seed(42 + i)
            
            train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False)
            
            # --- MODIFICATION START: Instantiate and print parameter count ---
            model = SingleDigitClassifier().to(DEVICE)
            
            print(f"[{name} - Run {i+1}] Parameter Count: {count_parameters(model):,}")
            
            is_ltn = config["type"] == "logic"
            if is_ltn:
                # After wrapping in an LTN Predicate, the underlying parameters are the same
                model = ltn.Predicate(LogitsToPredicate(model))
            # --- MODIFICATION END ---
            
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            for epoch in range(config["epochs"]):
                epoch_start_time = time.time(); model.train(); epoch_power, epoch_mem = [], []
                
                # TRAINING LOOP
                total_loss, correct_sum_train, correct_digit_train, total_train, total_sat = 0, 0, 0, 0, 0
                for operand_images, sum_label, digit_labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
                    optimizer.zero_grad()
                    if is_ltn:
                        And, Exists, Forall = config['params']['And'], config['params']['Exists'], config['params']['Forall']
                        d_1, d_2 = ltn.Variable("d_1", torch.arange(10, device=DEVICE)), ltn.Variable("d_2", torch.arange(10, device=DEVICE))
                        images_x, images_y, labels_z = ltn.Variable("x", operand_images[:, 0]), ltn.Variable("y", operand_images[:, 1]), ltn.Variable("z", sum_label)
                        sat_agg = Forall(ltn.diag(images_x, images_y, labels_z), Exists([d_1, d_2], And(model(images_x, d_1), model(images_y, d_2)), cond_vars=[d_1, d_2, labels_z], cond_fn=lambda d1, d2, z: torch.eq(d1.value + d2.value, z.value))).value
                        loss = 1.0 - sat_agg; loss.backward(); optimizer.step()
                        total_loss += loss.item(); total_sat += sat_agg.item()
                        with torch.no_grad():
                            out1 = torch.argmax(model.model.logits_model(operand_images[:, 0].to(DEVICE)), dim=1); out2 = torch.argmax(model.model.logits_model(operand_images[:, 1].to(DEVICE)), dim=1)
                            correct_sum_train += torch.sum(out1 + out2 == sum_label.to(DEVICE)).item()
                            correct_digit_train += torch.sum(out1 == digit_labels[:, 0].to(DEVICE)).item() + torch.sum(out2 == digit_labels[:, 1].to(DEVICE)).item()
                    else: # Baseline model
                        mse_loss, digit_range = torch.nn.MSELoss(), torch.arange(10, device=DEVICE).float()
                        img1, img2, sum_label_t, digits = operand_images[:, 0].to(DEVICE), operand_images[:, 1].to(DEVICE), sum_label.to(DEVICE), digit_labels.to(DEVICE)
                        out1, out2 = model(img1), model(img2)
                        probs1, probs2 = F.softmax(out1, dim=1), F.softmax(out2, dim=1)
                        pred_sum = probs1 @ digit_range + probs2 @ digit_range
                        loss = mse_loss(pred_sum, sum_label_t.float()); loss.backward(); optimizer.step()
                        total_loss += loss.item()
                        predicted_digit1, predicted_digit2 = torch.argmax(out1, dim=1), torch.argmax(out2, dim=1)
                        correct_sum_train += torch.sum(predicted_digit1 + predicted_digit2 == sum_label_t).item()
                        correct_digit_train += torch.sum(predicted_digit1 == digits[:, 0]).item() + torch.sum(predicted_digit2 == digits[:, 1]).item()
                    
                    total_train += sum_label.size(0)
                    if HAS_PYNVML and GPU_HANDLE:
                        epoch_power.append(pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000.0); epoch_mem.append(pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE).used / (1024**2))
                
                # EVALUATION LOOP
                model.eval(); correct_sum_test, correct_digits_test, total_test, test_loss_val, test_sat_val = 0, 0, 0, 0, 0
                with torch.no_grad():
                    for operand_images, sum_label, digit_labels, _ in test_loader:
                        if is_ltn:
                            d_1, d_2 = ltn.Variable("d_1", torch.arange(10, device=DEVICE)), ltn.Variable("d_2", torch.arange(10, device=DEVICE))
                            images_x, images_y, labels_z, digit_labels_t = ltn.Variable("x", operand_images[:, 0]), ltn.Variable("y", operand_images[:, 1]), ltn.Variable("z", sum_label), digit_labels.to(DEVICE)
                            sat_agg = Forall(ltn.diag(images_x, images_y, labels_z), Exists([d_1, d_2], And(model(images_x, d_1), model(images_y, d_2)), cond_vars=[d_1, d_2, labels_z], cond_fn=lambda d1, d2, z: torch.eq(d1.value + d2.value, z.value))).value
                            test_loss_val += (1.0 - sat_agg).item(); test_sat_val += sat_agg.item()
                            out1 = torch.argmax(model.model.logits_model(operand_images[:, 0].to(DEVICE)), dim=1); out2 = torch.argmax(model.model.logits_model(operand_images[:, 1].to(DEVICE)), dim=1)
                            correct_sum_test += torch.sum(out1 + out2 == sum_label.to(DEVICE)).item(); correct_digits_test += torch.sum(out1 == digit_labels_t[:, 0]).item() + torch.sum(out2 == digit_labels_t[:, 1]).item()
                        else: # Baseline model
                            mse_loss, digit_range = torch.nn.MSELoss(), torch.arange(10, device=DEVICE).float()
                            img1, img2, sum_label_t, digits = operand_images[:, 0].to(DEVICE), operand_images[:, 1].to(DEVICE), sum_label.to(DEVICE), digit_labels.to(DEVICE)
                            out1, out2 = model(img1), model(img2)
                            probs1, probs2 = F.softmax(out1, dim=1), F.softmax(out2, dim=1)
                            test_loss_val += mse_loss(probs1 @ digit_range + probs2 @ digit_range, sum_label_t.float()).item()
                            predicted_digit1, predicted_digit2 = torch.argmax(out1, dim=1), torch.argmax(out2, dim=1)
                            correct_sum_test += torch.sum(predicted_digit1 + predicted_digit2 == sum_label_t).item()
                            correct_digits_test += torch.sum(predicted_digit1 == digits[:, 0]).item() + torch.sum(predicted_digit2 == digits[:, 1]).item()
                        total_test += sum_label.size(0)

                # Save enhanced checkpoint with model state and metrics for this epoch
                checkpoint_path = f"checkpoints/epochs/{name.replace(' ', '_')}_run{i}_epoch{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss / len(train_loader),
                    'accuracy_sum': correct_sum_train / total_train,
                    'accuracy_digit': correct_digit_train / (2 * total_train),
                    'test_loss': test_loss_val / len(test_loader),
                    'test_accuracy_sum': correct_sum_test / total_test,
                    'test_accuracy_digit': correct_digits_test / (2 * total_test),
                    'sat': (total_sat / len(train_loader)) if is_ltn else None,
                    'test_sat': (test_sat_val / len(test_loader)) if is_ltn else None,
                    'epoch_time': time.time() - epoch_start_time,
                    'gpu_power_watt': np.mean(epoch_power) if epoch_power else 0,
                    'gpu_memory_mib': np.mean(epoch_mem) if epoch_mem else 0,
                }, checkpoint_path)

            print(f"Finished training run {i+1} for {name}. All enhanced epoch checkpoints saved.")

    if DEVICE.type == 'cuda' and HAS_PYNVML: pynvml.nvmlShutdown()
    print("\n" + "="*50 + "\nTRAINING SCRIPT COMPLETE\n" + "="*50)
    print("You can now run analyze.py to process the results.")