import os
import json
from glob import glob
from .PanoDataset import PanoDataset, PanoDataModule
from PIL import Image
import numpy as np


class PanimeDataset(PanoDataset):
    def load_split(self, mode):
        """Load the dataset split based on the mode (train or test)."""
        dataset_path = os.path.join(self.data_dir, 'dataset.json')
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Filter the dataset by mode
        return [item for item in dataset if item['split'] == mode]

    def scan_results(self, result_dir):
        """Scan results directory to collect processed panoramas."""
        results = glob(os.path.join(result_dir, '*.png'))
        results = [os.path.basename(r).split('.')[0] for r in results]
        return results

    def get_data(self, idx):
        """Load and return a single dataset entry."""
        data = self.data[idx].copy()

        # Basic data fields
        data['pano_id'] = os.path.splitext(os.path.basename(data['image']))[0]
        data['pano_path'] = os.path.join(self.data_dir, data['image'])

        # Load the image
        image = Image.open(data['pano_path']).convert('RGB')  # Convert to RGB
        image = np.array(image).astype('float32') / 127.5 - 1.0  # Normalize to [-1, 1]
        image = np.transpose(image, (2, 0, 1))  # Convert to channels-first format for PyTorch
        data['image'] = image

        # Pad or truncate the prompt
        max_prompt_length = 200  # Maximum string length
        prompt = data['prompt'][:max_prompt_length]  # Truncate
        prompt += ' ' * (max_prompt_length - len(prompt))  # Pad with spaces
        data['pano_prompt'] = prompt

        # Pad or truncate the tags
        max_tags_length = 15  # Maximum list length
        tags = data.get('tags', [])
        tags = tags[:max_tags_length] + [''] * (max_tags_length - len(tags))  # Pad with empty strings
        data['tags'] = tags

        # Pad or truncate the negative tags
        negative_tags = data.get('negative_tags', [])
        negative_tags = negative_tags[:max_tags_length] + [''] * (max_tags_length - len(negative_tags))
        data['negative_tags'] = negative_tags

        # Additional metadata
        data['mood'] = data.get('mood', '')
        data['lighting'] = data.get('lighting', '')

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
