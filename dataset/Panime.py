import os
import json
from glob import glob
from .PanoDataset import PanoDataset, PanoDataModule


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

        # Directly use the prompt from the dataset JSON
        data['pano_prompt'] = data['prompt']

        # Additional metadata
        data['mood'] = data.get('mood', '')
        data['tags'] = data.get('tags', [])
        data['negative_tags'] = data.get('negative_tags', [])
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
