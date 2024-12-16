from dataset.Panime import PanimeDataset

# Initialize the dataset for the "train" split
dataset = PanimeDataset(config = {
    "data_dir": "data/Panime",
    "fov": 90,
    "cam_sampler": "icosahedron",  # Or "horizon" depending on your test
    "pers_resolution": 256,
    "pano_height": 512,
    "uncond_ratio": 0.2,
    "batch_size": 4,
    "num_workers": 2,
    "rand_rot_img": False,
    "rand_flip": True,
    "gt_as_result": False,
    "horizon_layout": False,
    "manhattan_layout": False,
    "layout_cond_type": None,
    "repeat_predict": 10,
}, mode="train")

# Check the number of samples
print(f"Number of samples in train dataset: {len(dataset)}")

# Load and inspect a single sample
sample = dataset[0]
print("Sample data:")
print(f"Pano ID: {sample['pano_id']}")
print(f"Prompt: {sample['pano_prompt']}")
print(f"Image shape: {sample['image'].shape}")
