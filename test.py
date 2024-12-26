import torch
from dataset import PanimeDataModule  # Adjust this import as per your actual file structure

def test_last_batch():
    # Set up your configuration (replace with actual config or hardcode as needed)
    config = {
        'data_dir': 'data/Panime',  # Replace with your actual dataset directory
        'batch_size': 4,
        'num_workers': 1,  # Adjust as per your system setup
    }

    # Initialize the data module and dataloaders
    data_module = PanimeDataModule(**config)
    data_module.setup(stage='fit')  # Initializes the train dataset, etc.
    
    # Get the train dataloader
    train_loader = data_module.train_dataloader()

    # Iterate through the dataloader to fetch the last batch
    last_batch = None
    for batch_idx, batch in enumerate(train_loader):
        last_batch = batch
        print(f"Batch {batch_idx + 1}:")
        print(f"  Image batch shape: {batch['image'].shape}")
        print(f"  Pano prompt: {batch['pano_prompt']}")
        
    if last_batch:
        print("\nLast batch fetched:")
        print(f"  Image batch shape: {last_batch['image'].shape}")
        print(f"  Pano prompt: {last_batch['pano_prompt']}")
    else:
        print("No batches were fetched.")

if __name__ == '__main__':
    test_last_batch()
