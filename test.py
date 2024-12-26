from dataset.Panime import PanimeDataset, PanimeDataModule

def test_loader():
    # Mock configuration
    config = {
        'data_dir': 'data/Panime',  # Path to your dataset
        'fov': 90,
        'cam_sampler': 'icosahedron',  # Options: 'icosahedron', 'horizon'
        'pers_resolution': 256,
        'pano_height': 512,
        'uncond_ratio': 0.2,
        'batch_size': 2,
        'num_workers': 0,  # Set to 0 for testing to avoid parallelism issues
        'result_dir': None,  # Set if you have result predictions
        'rand_rot_img': False,
        'rand_flip': True,
        'gt_as_result': False,
        'horizon_layout': False,
        'manhattan_layout': False,
        'layout_cond_type': None,  # Set to a specific type if needed
        'repeat_predict': 10,  # For prediction mode
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
        print(f"  Prompts: {batch['pano_prompt']}")
        print(f"  Moods: {batch['mood']}")
        print(f"  Tags batch: {batch['tags']}")
        print(f"  Negative Tags batch: {batch['negative_tags']}")
        print(f"  Lighting batch: {batch['lighting']}")
        print("-" * 30)
        break  # Print only one batch for simplic

if __name__ == "__main__":
    test_loader()
