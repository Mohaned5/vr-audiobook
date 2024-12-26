from Panime import PanimeDataset, PanimeDataModule

def test_loader():
    # Mock configuration
    config = {
        'data_dir': 'data/Panime',
        'result_dir': None,  # Set this if you have a result directory
        'fov': 90,
        'pers_resolution': 256,
        'pano_height': 512,
        'batch_size': 2,
        'num_workers': 0,
    }

    # Instantiate and test the dataset
    print("Testing PanimeDataset...")
    dataset = PanimeDataset(config, mode='train')

    print(f"Total samples in train dataset: {len(dataset)}")

    # Print a few samples from the dataset
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image Shape: {sample['image'].shape}")
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Mood: {sample['mood']}")
        print(f"  Tags: {sample['tags']}")
        print(f"  Negative Tags: {sample['negative_tags']}")
        print(f"  Lighting: {sample['lighting']}")
        print("-" * 30)

    # Instantiate and test the data module
    print("\nTesting PanimeDataModule...")
    data_module = PanimeDataModule(**config)
    data_module.setup(stage='fit')

    train_loader = data_module.train_dataloader()
    print(f"Number of batches in train loader: {len(train_loader)}")

    # Fetch and print a batch
    for batch in train_loader:
        print("Batch example:")
        print(f"  Image batch shape: {batch['image'].shape}")
        print(f"  Prompts: {batch['prompt']}")
        print("-" * 30)
        break  # Just show one batch for simplicity

if __name__ == "__main__":
    test_loader()
