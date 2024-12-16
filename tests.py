from dataset.Panime import PanimeDataset

# Initialize the dataset for the "train" split
dataset = PanimeDataset(config={"data_dir": "data/Panime"}, mode="train")

# Check the number of samples
print(f"Number of samples in train dataset: {len(dataset)}")

# Load and inspect a single sample
sample = dataset[0]
print("Sample data:")
print(f"Pano ID: {sample['pano_id']}")
print(f"Prompt: {sample['pano_prompt']}")
print(f"Image shape: {sample['image'].shape}")
