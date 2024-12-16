import os
from .PanoDataset import PanoDataset, PanoDataModule
from PIL import Image
import numpy as np

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