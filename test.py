"""
test.py
Enhanced test script for the updated PanimeDataset and PanimeDataModule.
Includes checks to ensure images loaded from the dataset match the ones on disk.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataset.Panime import PanimeDataset, PanimeDataModule

def visualize_sample(sample, index):
    """
    Display a sample image with its pano_prompt for manual verification.
    """
    image = sample['image']
    pano_prompt = sample['pano_prompt']

    # Denormalize image and convert to HWC format
    image = (image + 1.0) * 127.5  # From [-1, 1] to [0, 255]
    image = image.astype('uint8').transpose(1, 2, 0)  # Channels first to HWC

    # Plot the image with the prompt
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Sample {index}: {pano_prompt}", wrap=True, fontsize=10)
    plt.show()

def validate_image_match(sample):
    """
    Compare the loaded image in the dataset to the one on disk to ensure correctness.
    """
    pano_path = sample['pano_path']
    # Load image directly from disk
    disk_image = Image.open(pano_path).convert('RGB')
    disk_array = np.array(disk_image)

    # Denormalize the dataset image for comparison
    dataset_image = (sample['image'] + 1.0) * 127.5  # From [-1, 1] to [0, 255]
    dataset_image = dataset_image.astype('uint8').transpose(1, 2, 0)

    # Compare the two images
    if not np.array_equal(disk_array, dataset_image):
        print(f"Mismatch detected for image: {pano_path}")
        return False
    return True

def test_loader():
    # Mock configuration
    config = {
        'data_dir': 'data/Panime',     # Path to your dataset
        'fov': 90,
        'cam_sampler': 'icosahedron',
        'pers_resolution': 256,
        'pano_height': 512,
        'uncond_ratio': 0.2,
        'batch_size': 2,
        'num_workers': 0,  # For testing, often safer to keep this 0
        'result_dir': None,
        'rand_rot_img': False,
        'rand_flip': True,
        'gt_as_result': False,
        'horizon_layout': False,
        'manhattan_layout': False,
        'layout_cond_type': None,
        'repeat_predict': 10
    }

    # Instantiate and test the dataset directly
    print("Testing PanimeDataset...")
    dataset = PanimeDataset(config, mode='train')
    print(f"Total samples in train dataset: {len(dataset)}")

    # Print a few samples from the dataset
    for i in range(10):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image Shape: {sample['image'].shape}")
        print(f"  Pano Prompt: {sample['pano_prompt']}")
        print(f"  Image Path: {sample['pano_path']}")

        # Validate the loaded image matches the disk image
        if validate_image_match(sample):
            print(f"  Sample {i} image matches the disk file.")
        else:
            print(f"  Sample {i} image does NOT match the disk file.")

        # Optional: Visualize the sample
        visualize_sample(sample, i)
        print("-" * 30)

    # Instantiate and test the data module
    print("\nTesting PanimeDataModule...")
    data_module = PanimeDataModule(**config)
    data_module.setup(stage='fit')  # Prepares train_dataset, val_dataset, etc.

    train_loader = data_module.train_dataloader()
    print(f"Number of batches in train loader: {len(train_loader)}")

    # Fetch and print a batch
    i = 0
    for batch in train_loader:
        print("Batch example:")
        print(f"  Image batch shape: {batch['image'].shape}")
        print(f"  Pano prompts in batch: {batch['pano_prompt']}")
        print("-" * 30)
        if i == 5:
            break  # Print only one batch for simplicity
        i += 1

if __name__ == "__main__":
    test_loader()
