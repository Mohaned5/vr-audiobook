import os
import json
from glob import glob
from .PanoDataset import PanoDataset, PanoDataModule
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from utils.pano_collate import pano_collate_fn

class PanimeDataset(PanoDataset):
    """
    Custom dataset that merges mood, lighting, tags, negative_tags, and original
    prompt into one textual field (pano_prompt) for Stable Diffusion–style training.
    """

    def load_split(self, mode):
        """Load the dataset split based on the mode (train/val/test/predict)."""
        dataset_path = os.path.join(self.data_dir, 'dataset.json')
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Filter the dataset by the given split (e.g., "train", "val", "test")
        return [item for item in dataset if item['split'] == mode]

    def scan_results(self, result_dir):
        """Scan results directory to collect processed panoramas (if needed)."""
        results = glob(os.path.join(result_dir, '*.png'))
        results = [os.path.basename(r).split('.')[0] for r in results]
        return results

    def unify_text_fields(self, data):
        """
        Merge mood, lighting, tags, negative_tags, and original prompt into one string.
        If you prefer to keep negative_tags separate for a negative prompt, you can
        do so, but here's an all-in-one approach for simplicity.
        """
        prompt = data.get("prompt", "")
        mood = data.get("mood", "")
        lighting = data.get("lighting", "")
        tags = data.get("tags", [])
        negative_tags = data.get("negative_tags", [])

        mood_str = f"Mood: {mood}. " if mood else ""
        lighting_str = f"Lighting: {lighting}. " if lighting else ""
        tags_str = f"Tags: {', '.join(tags)}. " if tags else ""
        # neg_tags_str = f"Negative tags: {', '.join(negative_tags)}. " if negative_tags else ""

        # Combine them into a single string
        combined_prompt = f"{mood_str}{lighting_str}{tags_str}{prompt}"
        return combined_prompt

    def get_data(self, idx):
        """Load and return a single dataset entry with unified text fields."""
        data = self.data[idx].copy()

        # Basic data fields
        data['pano_id'] = os.path.splitext(os.path.basename(data['image']))[0]
        data['pano_path'] = os.path.join(self.data_dir, data['image'])

        # Load the image
        image = Image.open(data['pano_path']).convert('RGB')  # Convert to RGB
        width, height = image.size

        # Check if the image size matches the expected dimensions (2048x1024)
        if (width, height) != (2048, 1024):
            print(f"WARNING: Image {data['pano_path']} has size {width}x{height}, expected 2048x1024")

        # Convert image to NumPy array, normalize to [-1, 1], channels-first
        image = np.array(image).astype('float32') / 127.5 - 1.0
        image = np.transpose(image, (2, 0, 1))
        data['image'] = image

        data['pano_prompt'] = self.unify_text_fields(data)

        return data


class PanimeDataModule(PanoDataModule):

    def __init__(
            self,
            data_dir: str = 'data/Panime',
            *args,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.dataset_cls = PanimeDataset  

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            collate_fn=pano_collate_fn  
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            collate_fn=pano_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            collate_fn=pano_collate_fn
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            collate_fn=pano_collate_fn
        )
