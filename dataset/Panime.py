import os
from .PanoDataset import PanoDataset, PanoDataModule
from PIL import Image
import numpy as np
# File: dataset/Panime.py
import lightning as L
from torch.utils.data import DataLoader

class PanimeDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 4, pano_height: int = 512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pano_height = pano_height

    @staticmethod
    def custom_collate_fn(batch):
        # Ensure all images and pano tensors are stacked
        images = torch.stack([sample['image'] for sample in batch])
        pano = torch.stack([sample['pano'] for sample in batch])
        
        # Collect metadata (variable-length fields)
        mood = [sample['mood'] for sample in batch]  # List of strings
        tags = [sample['tags'] for sample in batch]  # List of lists
        negative_tags = [sample['negative_tags'] for sample in batch]  # List of lists
        lighting = [sample['lighting'] for sample in batch]  # List of strings
        
        # Ensure cameras are properly batched
        cameras = {key: torch.stack([sample['cameras'][key] for sample in batch]) for key in batch[0]['cameras']}
        
        return {
            'images': images,          # Tensor
            'pano': pano,              # Tensor
            'mood': mood,              # List[str]
            'tags': tags,              # List[List[str]]
            'negative_tags': negative_tags,  # List[List[str]]
            'lighting': lighting,      # List[str]
            'cameras': cameras,        # Dict[str, Tensor]
        }

    def setup(self, stage=None):
        dataset_config = {"data_dir": self.data_dir, "pano_height": self.pano_height}
        
        if stage in ("fit", None):
            self.train_dataset = PanimeDataset(config=dataset_config, mode="train")

        if stage in ("fit", "validate", None):
            self.val_dataset = PanimeDataset(config=dataset_config, mode="val")

        if stage == "test":
            self.test_dataset = PanimeDataset(config=dataset_config, mode="test")

        if stage == "predict":
            self.predict_dataset = PanimeDataset(config=dataset_config, mode="predict")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn
        )


class PanimeDataset(PanoDataset):
    def load_split(self, mode):
        """
        Load the dataset split from the JSON file based on the mode (train/val/test).
        Validation split does not include images.
        """
        import json

        # Load dataset JSON
        dataset_path = os.path.join(self.config['data_dir'], "dataset.json")
        with open(dataset_path, "r") as f:
            all_data = json.load(f)

        # Filter by split
        if mode == "val":
            split_data = [item for item in all_data if item["split"] == mode]
            for entry in split_data:
                entry["image"] = None  # Remove image field for validation
        else:
            split_data = [item for item in all_data if item["split"] == mode]

        return split_data


    def scan_results(self, result_dir):
        """
        Scan the result directory for existing predictions.
        Args:
            result_dir (str): The path to the result directory.
        Returns:
            list: A list of existing result identifiers.
        """
        results = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(result_dir) if f.endswith(".png")]
        return results

    def get_data(self, idx):
        # Retrieve the data sample
        item = self.data[idx].copy()

        # Construct the pano path
        item['pano_id'] = f"panime_{idx:06d}"
        item['pano_path'] = os.path.join(self.config["data_dir"], item["image"])

        # Load the image
        if os.path.exists(item['pano_path']):
            pano_image = Image.open(item['pano_path']).convert("RGB")
            item['image'] = np.array(pano_image)
        else:
            print(f"Image not found: {item['pano_path']}")
            item['image'] = None  # Handle missing images gracefully

        # Use the 'prompt' field as pano_prompt
        item['pano_prompt'] = item.get('prompt', '')  # Default to an empty string if 'prompt' is missing
        item['mood'] = item.get('mood', '')  # Load mood, default to empty string
        item['tags'] = item.get('tags', [])  # Load tags, default to an empty list
        item['negative_tags'] = item.get('negative_tags', [])  # Load negative tags, default to empty list
        item['lighting'] = item.get('lighting', '') 

        return item

class PanimeDataModule(PanoDataModule):
    def __init__(self, data_dir="data/Panime", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams["data_dir"] = data_dir
        self.dataset_cls = PanimeDataset