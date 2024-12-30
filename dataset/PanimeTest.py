import os
import json
import random
import numpy as np
import torch
from glob import glob
from PIL import Image
import cv2
import lightning as L

from einops import rearrange
from abc import abstractmethod
from utils.pano import Equirectangular, random_sample_camera, horizon_sample_camera, icosahedron_sample_camera
from external.Perspective_and_Equirectangular import mp2e
from .PanoDataset import PanoDataset, PanoDataModule, get_K_R


class PanimeDataset(PanoDataset):

    def load_split(self, mode):
        split_file = os.path.join(self.data_dir, f"{mode}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Cannot find JSON split file: {split_file}")

        with open(split_file, 'r') as f:
            all_data = json.load(f)

        new_data = []
        for sample in all_data:
            # Create a unique pano_id from the file name (or any logic you prefer)
            pano_filename = os.path.basename(sample["pano"])
            pano_id = os.path.splitext(pano_filename)[0]

            # We'll store everything we need in a single dict
            entry = {
                "pano_id": pano_id,
                "pano_path": os.path.join(self.data_dir, sample["pano"]),
                "pano_prompt": sample.get("pano_prompt", ""),
                "images_paths": [os.path.join(self.data_dir, img) for img in sample["images"]],
                "prompts": sample["prompts"],
                "cameras_data": sample["cameras"]
            }
            new_data.append(entry)
        return new_data

    def scan_results(self, result_dir):
        """
        If your results are saved under result_dir/<pano_id>/pano.png,
        then we can detect them by scanning all subfolders in result_dir.
        """
        folder_paths = glob(os.path.join(result_dir, '*'))
        # Extract folder names that might be valid pano_ids
        results_ids = {
            os.path.basename(p)
            for p in folder_paths
            if os.path.isdir(p)
        }
        return results_ids

    def get_data(self, idx):
        data = self.data[idx].copy()

        # Construct the sample to return
        # 1) If in predict mode with repeated sampling
        if self.mode == 'predict' and self.config['repeat_predict'] > 1:
            data['pano_id'] = f"{data['pano_id']}_{data['repeat_id']:06d}"

        # 2) Possibly apply unconditioned training ratio
        if self.mode == 'train' and self.result_dir is None and random.random() < self.config['uncond_ratio']:
            data['pano_prompt'] = ""
            data['prompts'] = [""] * len(data['prompts'])

        # 3) Build 'cameras' dict in the format PanFusion expects
        #    Here, 'FoV', 'theta', 'phi' might be 2D lists: e.g. shape [1, M]
        cam_data = data['cameras_data']
        FoV = np.array(cam_data['FoV'][0], dtype=np.float32)
        theta = np.array(cam_data['theta'][0], dtype=np.float32)
        phi = np.array(cam_data['phi'][0], dtype=np.float32)

        cameras = {
            'height': 512,
            'width': 512,
            'FoV': FoV,
            'theta': theta,
            'phi': phi,
        }

        # Compute intrinsics & extrinsics
        Ks, Rs = [], []
        for f, t, p in zip(FoV, theta, phi):
            K, R = get_K_R(f, t, p, self.config['pers_resolution'], self.config['pers_resolution'])
            Ks.append(K)
            Rs.append(R)
        cameras['K'] = np.stack(Ks).astype(np.float32)
        cameras['R'] = np.stack(Rs).astype(np.float32)

        # 4) Save everything the pipeline expects
        data['prompt'] = data['prompts']
        data['cameras'] = cameras
        data['height'] = self.config['pano_height']
        data['width'] = self.config['pano_height'] * 2  # typical equirect (2:1 ratio)

        # 5) Provide the path for the pano
        if self.mode != 'predict':
            data['pano_path'] = data['pano_path']

        # 6) If there's a results directory, set path for predicted pano
        if self.result_dir is not None:
            data['pano_pred_path'] = os.path.join(self.result_dir, data['pano_id'], 'pano.png')

        return data


class PanimeDataModule(PanoDataModule):
    """
    The custom DataModule for your dataset, similar to Matterport3D,
    but referencing PanimeDataset as the dataset class.
    """

    def __init__(
        self,
        data_dir: str = 'data/Panime',
        # All other arguments same as base class...
        # You can override or add new ones if needed.
        **kwargs
    ):
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        # We set the dataset class to PanimeDataset
        self.dataset_cls = PanimeDataset

    def setup(self, stage=None):
        # Exactly like base, but with PanimeDataset
        if stage in ('fit', None):
            self.train_dataset = self.dataset_cls(self.hparams, mode='train')
        if stage in ('fit', 'validate', None):
            self.val_dataset = self.dataset_cls(self.hparams, mode='val')
        if stage in ('test', None):
            self.test_dataset = self.dataset_cls(self.hparams, mode='test')
        if stage in ('predict', None):
            self.predict_dataset = self.dataset_cls(self.hparams, mode='predict')
