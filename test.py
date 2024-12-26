"""
test.py
Example test script for the updated PanimeDataset and PanimeDataModule.
"""

import os
from dataset.Panime import PanimeDataset, PanimeDataModule

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
        # If you kept separate fields, you could print them too:
        # print(f"  Mood: {sample.get('mood')}")
        # print(f"  Tags: {sample.get('tags')}")
        # print(f"  Negative Tags: {sample.get('negative_tags')}")
        # print(f"  Lighting: {sample.get('lighting')}")
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
        # 'pano_prompt' is the single merged string for each sample
        print(f"  Pano prompts in batch: {batch['pano_prompt']}")
        print("-" * 30)
        if i == 5:
            break  # Print only one batch for simplicity
        i += 1

if __name__ == "__main__":
    test_loader()
