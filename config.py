import torchvision
from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np

# 1. CONFIGURATION (IMPORTANT: Must match train.py)
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_RUNS = 1
N_EPOCHS_BASELINE = 1
N_EPOCHS_LOGIC = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.001
N_PROBE_REPEATS = 2
N_EPOCHS_VAE_PER_PROBE = 21

# =============================================================================
# 2. SETUP: DEVICE, DIRECTORIES, AND HELPERS
# =============================================================================
print(f"Using device: {DEVICE}")

def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# =============================================================================
# 3. DATA HANDLING and MODELS (Must be identical to train.py)
# =============================================================================
def get_mnist_dataset_for_digits_addition():
    n_train_examples, n_test_examples, n_operands = 30000, 5000, 2
    mnist_train = torchvision.datasets.MNIST("./datasets/", train=True, download=True, transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST("./datasets/", train=False, download=True, transform=torchvision.transforms.ToTensor())
    train_imgs, train_labels = mnist_train.data.float() / 255.0, mnist_train.targets
    imgs_operand_train = [train_imgs[i * n_train_examples:(i + 1) * n_train_examples] for i in range(n_operands)]
    labels_operand_train = [train_labels[i * n_train_examples:(i + 1) * n_train_examples] for i in range(n_operands)]
    test_imgs, test_labels = mnist_test.data.float() / 255.0, mnist_test.targets
    imgs_operand_test = [test_imgs[i * n_test_examples:(i + 1) * n_test_examples] for i in range(n_operands)]
    labels_operand_test = [test_labels[i * n_test_examples:(i + 1) * n_test_examples] for i in range(n_operands)]
    train_set = [torch.stack(imgs_operand_train, dim=1).unsqueeze(2), labels_operand_train[0] + labels_operand_train[1], torch.stack(labels_operand_train, dim=1), torch.stack(imgs_operand_train, dim=1)]
    test_set = [torch.stack(imgs_operand_test, dim=1).unsqueeze(2), labels_operand_test[0] + labels_operand_test[1], torch.stack(labels_operand_test, dim=1), torch.stack(imgs_operand_test, dim=1)]
    return train_set, test_set

class DataLoader(object):
    def __init__(self, fold: list, batch_size: int = 1, shuffle: bool = True):
        self.fold, self.batch_size, self.shuffle = fold, batch_size, shuffle
        self.num_samples = self.fold[0].shape[0]
    def __len__(self): return int(np.ceil(self.num_samples / self.batch_size))
    def __iter__(self):
        idxlist = list(range(self.num_samples))
        if self.shuffle: np.random.shuffle(idxlist)
        for _, start_idx in enumerate(range(0, self.num_samples, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            yield (self.fold[0][idxlist[start_idx:end_idx]], self.fold[1][idxlist[start_idx:end_idx]], self.fold[2][idxlist[start_idx:end_idx]], self.fold[3][idxlist[start_idx:end_idx]])

class MNISTFeatureExtractor(nn.Module):
    def __init__(self):
        super(MNISTFeatureExtractor, self).__init__(); self.conv1 = nn.Conv2d(1, 6, 5); self.conv2 = nn.Conv2d(6, 16, 5); self.fc1 = nn.Linear(16 * 4 * 4, 100)
        self.relu, self.maxpool, self.tanh, self.batch_norm = nn.ReLU(), nn.MaxPool2d(2, 2), nn.Tanh(), nn.BatchNorm1d(100)
    def forward(self, x: torch.Tensor):
        activations = {}; x = x.unsqueeze(1) if x.dim() == 3 else x; out = self.maxpool(self.relu(self.conv1(x))); activations['conv1'] = out; out = self.maxpool(self.relu(self.conv2(out))); activations['conv2'] = out
        out = torch.flatten(out, 1); out = self.fc1(out); activations['fc1_pre_act'] = out; out = self.tanh(self.batch_norm(out)); activations['features'] = out
        return activations

class SingleDigitClassifier(nn.Module):
    def __init__(self):
        super(SingleDigitClassifier, self).__init__(); self.feature_extractor = MNISTFeatureExtractor()
        self.classifier_head = nn.Sequential(nn.Linear(100, 84), nn.BatchNorm1d(84), nn.Tanh(), nn.Linear(84, 10))
    def forward(self, x: torch.Tensor): return self.classifier_head(self.feature_extractor(x)['features'])

class LogitsToPredicate(nn.Module):
    def __init__(self, logits_model: nn.Module):
        super(LogitsToPredicate, self).__init__(); self.logits_model = logits_model; self.softmax = nn.Softmax(dim=1)
    def forward(self, x: torch.Tensor, d: torch.Tensor):
        return torch.gather(self.softmax(self.logits_model(x.to(DEVICE))), 1, d.long())
