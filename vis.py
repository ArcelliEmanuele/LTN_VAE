import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# =============================================================================
# This code is reused from your provided scripts to ensure consistency.
# =============================================================================

def get_mnist_dataset_for_digits_addition() -> Tuple[list, list]:
    """
    Loads and prepares the MNIST dataset for the digit addition task.
    This function is taken directly from your 'analyze.py' script.
    """
    # Using smaller numbers for a quick demonstration
    n_train_examples, n_test_examples, n_operands = 5000, 1000, 2
    
    # Download and load the MNIST dataset
    print("Downloading MNIST dataset (if not already present)...")
    mnist_train = torchvision.datasets.MNIST(
        "./datasets/", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    mnist_test = torchvision.datasets.MNIST(
        "./datasets/", train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    print("Dataset download complete.")
    
    # Prepare training data by pairing images and labels
    train_imgs, train_labels = mnist_train.data / 255.0, mnist_train.targets
    imgs_operand_train = [train_imgs[i * n_train_examples:(i + 1) * n_train_examples] for i in range(n_operands)]
    labels_operand_train = [train_labels[i * n_train_examples:(i + 1) * n_train_examples] for i in range(n_operands)]
    
    # Prepare test data similarly
    test_imgs, test_labels = mnist_test.data / 255.0, mnist_test.targets
    imgs_operand_test = [test_imgs[i * n_test_examples:(i + 1) * n_test_examples] for i in range(n_operands)]
    labels_operand_test = [test_labels[i * n_test_examples:(i + 1) * n_test_examples] for i in range(n_operands)]

    # Structure the data into a list of tensors as your scripts expect
    # [operand_images, sum_labels, digit_labels, raw_images]
    train_set = [
        torch.stack(imgs_operand_train, dim=1).unsqueeze(2), 
        labels_operand_train[0] + labels_operand_train[1], 
        torch.stack(labels_operand_train, dim=1), 
        torch.stack(imgs_operand_train, dim=1)
    ]
    test_set = [
        torch.stack(imgs_operand_test, dim=1).unsqueeze(2), 
        labels_operand_test[0] + labels_operand_test[1], 
        torch.stack(labels_operand_test, dim=1), 
        torch.stack(imgs_operand_test, dim=1)
    ]
    
    return train_set, test_set

class DataLoader(object):
    """
    A custom DataLoader to handle the specific data structure.
    This class is taken directly from your 'analyze.py' script.
    """
    def __init__(self, fold: list, batch_size: int = 1, shuffle: bool = True):
        self.fold = fold
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = self.fold[0].shape[0]

    def __len__(self) -> int:
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self) -> Tuple[torch.Tensor, ...]:
        idxlist = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(idxlist)
        
        for _, start_idx in enumerate(range(0, self.num_samples, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = idxlist[start_idx:end_idx]
            
            # Yields a tuple for each batch
            yield (
                self.fold[0][batch_indices], 
                self.fold[1][batch_indices], 
                self.fold[2][batch_indices], 
                self.fold[3][batch_indices]
            )

# =============================================================================
# Main script to load and visualize examples
# =============================================================================

if __name__ == '__main__':
    print("\nLoading the MNIST dataset for the addition task...")
    # 1. Get the dataset using the function from your scripts
    train_set, _ = get_mnist_dataset_for_digits_addition()
    
    print("Creating a DataLoader...")
    # 2. Create a DataLoader instance with a batch size of 8
    #    Shuffle is set to False to get consistent examples for demonstration
    example_loader = DataLoader(train_set, batch_size=8, shuffle=False)
    
    print("Fetching one batch of data...")
    # 3. Get the first batch of data from the loader
    # The 'iter' function makes the loader an iterator, and 'next' fetches the first item
    first_batch = next(iter(example_loader))
    
    # Unpack the batch into its components
    operand_images, sum_labels, digit_labels, _ = first_batch
    
    # 4. Define how many examples from the batch you want to see
    num_examples_to_show = 5
    print(f"\n--- Displaying the first {num_examples_to_show} examples from the batch ---\n")

    # 5. Loop through the examples and display them
    for i in range(num_examples_to_show):
        # Get the data for the i-th example in the batch
        img1 = operand_images[i, 0].squeeze() # First operand image
        img2 = operand_images[i, 1].squeeze() # Second operand image
        
        digit1_label = digit_labels[i, 0].item() # Label for the first digit
        digit2_label = digit_labels[i, 1].item() # Label for the second digit
        
        sum_label = sum_labels[i].item() # The label for the sum
        
        # --- Print the information ---
        print(f"Example {i+1}:")
        print(f"  - First Digit: {digit1_label}")
        print(f"  - Second Digit: {digit2_label}")
        print(f"  - Correct Sum: {sum_label}")
        
        # --- Visualize the images using Matplotlib ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        
        fig.suptitle(f'Example {i+1}: The task is to learn {digit1_label} + {digit2_label} = {sum_label}', fontsize=14)
        
        ax1.imshow(img1, cmap='gray')
        ax1.set_title(f"Image of Digit: {digit1_label}")
        ax1.axis('off') # Hide axes
        
        ax2.imshow(img2, cmap='gray')
        ax2.set_title(f"Image of Digit: {digit2_label}")
        ax2.axis('off') # Hide axes
        
        fig.savefig('ciao.png')
        plt.show()