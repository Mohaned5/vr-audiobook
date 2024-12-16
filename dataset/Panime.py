import os
from .PanoDataset import PanoDataset, PanoDataModule

class PanimeDataset(PanoDataset):
    def load_split(self, mode):
        """
        Load the dataset split from the JSON file based on the mode (train/val/test).
        Validation split does not include images.
        """
        import json

        # Load dataset JSON
        dataset_path = os.path.join(self.config['data_dir'], "panime_dataset.json")
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
        """
        Get a single data sample by index.
        Args:
            idx (int): The index of the data sample to retrieve.
        Returns:
            dict: A dictionary containing the sample's data.
        """
        # Retrieve the sample data
        item = self.data[idx].copy()
        
        # Add pano_id (unique identifier)
        item["pano_id"] = f"panime_{idx:06d}"

        # Add pano_path (path to the panoramic image)
        item["pano_path"] = os.path.join(self.config["data_dir"], item["image"])

        # Add pano_prompt (prompt for conditioning)
        item["pano_prompt"] = item["prompt"]

        return item

class PanimeDataModule(PanoDataModule):
    def __init__(self, data_dir="path/to/dataset", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams["data_dir"] = data_dir
        self.dataset_cls = PanimeDataset