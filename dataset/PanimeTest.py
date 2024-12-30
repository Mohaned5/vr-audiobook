import os
import json
import random
import numpy as np
import torch
from glob import glob
import cv2
from PIL import Image
import lightning as L

from einops import rearrange
from abc import abstractmethod

from utils.pano import Equirectangular, random_sample_camera, horizon_sample_camera, icosahedron_sample_camera
from external.Perspective_and_Equirectangular import mp2e
from .PanoDataset import PanoDataset, PanoDataModule, get_K_R


############################
# PanimeDataset
############################
class PanimeDataset(PanoDataset):
    """
    Loads panoramas at 2048×1024 from disk, then downscales to 1024×512.
    Loads perspective images at 512×512 from disk, then downscales to 256×256.
    """

    def load_split(self, mode):
        split_file = os.path.join(self.data_dir, f"{mode}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Cannot find JSON split file: {split_file}")

        with open(split_file, 'r') as f:
            all_data = json.load(f)

        new_data = []
        for sample in all_data:
            pano_filename = os.path.basename(sample["pano"])
            pano_id = os.path.splitext(pano_filename)[0]

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
        folder_paths = glob(os.path.join(result_dir, '*'))
        results_ids = {
            os.path.basename(p)
            for p in folder_paths
            if os.path.isdir(p)
        }
        return results_ids

    def get_data(self, idx):
        data = self.data[idx].copy()

        # Repeat logic for predict mode
        if self.mode == 'predict' and self.config['repeat_predict'] > 1:
            data['pano_id'] = f"{data['pano_id']}_{data['repeat_id']:06d}"

        # Possibly apply unconditioned training ratio
        if self.mode == 'train' and self.result_dir is None and random.random() < self.config['uncond_ratio']:
            data['pano_prompt'] = ""
            data['prompts'] = [""] * len(data['prompts'])

        # Parse camera data
        cam_data = data['cameras_data']
        FoV = np.array(cam_data['FoV'][0], dtype=np.float32)
        theta = np.array(cam_data['theta'][0], dtype=np.float32)
        phi = np.array(cam_data['phi'][0], dtype=np.float32)

        # We'll store the perspective images in memory at 256×256
        cameras = {
            "height": 256,
            "width": 256,
            "FoV": FoV,
            "theta": theta,
            "phi": phi,
        }

        # Intrinsics/extrinsics for 256×256
        Ks, Rs = [], []
        for f, t, p in zip(FoV, theta, phi):
            K, R = get_K_R(f, t, p, 256, 256)
            Ks.append(K)
            Rs.append(R)
        cameras["K"] = np.stack(Ks).astype(np.float32)
        cameras["R"] = np.stack(Rs).astype(np.float32)

        data["prompt"] = data["prompts"]
        data["cameras"] = cameras
        # The panorama in memory: 1024×512
        data["height"] = 512
        data["width"] = 1024

        # -----------------------
        # 1) Load & downscale Pano
        # -----------------------
        if self.mode != "predict":
            pano = cv2.imread(data["pano_path"])  # 2048×1024 on disk
            pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
            # Downscale to 1024×512
            pano_resized = cv2.resize(pano, (1024, 512), interpolation=cv2.INTER_AREA)
            # Convert to shape [C, H, W], normalized to -1..1 if needed
            pano_resized = pano_resized.astype(np.float32)
            if self.result_dir is None:
                pano_resized = (pano_resized / 127.5) - 1.0
            pano_resized = rearrange(pano_resized, "h w c -> 1 c h w")
            data["pano"] = pano_resized

        # ----------------------------------
        # 2) Load & downscale Perspective IMGs
        # ----------------------------------
        # If your pipeline actually needs them as "GT" or references
        # (the default PanFusion might rely on equirect->pers transform,
        # but if you do want them, here's how to load & store them).
        pers_list = []
        for img_path in data["images_paths"]:
            pers_img = cv2.imread(img_path)  # 512×512 on disk
            if pers_img is None:
                # skip or raise error if file not found
                continue
            pers_img = cv2.cvtColor(pers_img, cv2.COLOR_BGR2RGB)
            # Downscale to 256×256
            pers_img = cv2.resize(pers_img, (256, 256), interpolation=cv2.INTER_AREA)
            pers_img = pers_img.astype(np.float32)
            # If not predicting, we can normalize to -1..1
            if self.mode == "train" and self.result_dir is None:
                pers_img = (pers_img / 127.5) - 1.0
            pers_list.append(pers_img)

        if pers_list:
            pers_array = np.stack(pers_list)  # shape [N, 256, 256, 3]
            # reorder to [N, 3, 256, 256]
            pers_array = rearrange(pers_array, "n h w c -> n c h w")
            data["images"] = pers_array

        # If there's a results directory, set path for predicted pano
        if self.result_dir is not None:
            data["pano_pred_path"] = os.path.join(self.result_dir, data["pano_id"], "pano.png")

        return data


############################
# PanimeDataModule
############################
class PanimeDataModule(PanoDataModule):
    """
    A stripped-down data module focusing on training (and optionally predict),
    with in-memory downscale:
      - Panorama: 2048×1024 -> 1024×512
      - Perspective: 512×512 -> 256×256
    """

    def __init__(
        self,
        data_dir: str = 'data/Panime',
        **kwargs
    ):
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        self.dataset_cls = PanimeDataset

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.train_dataset = self.dataset_cls(self.hparams, mode='train')

        # Comment out val/test if you don't need them
        # if stage in ('fit', 'validate', None):
        #     self.val_dataset = self.dataset_cls(self.hparams, mode='val')

        # if stage in ('test', None):
        #     self.test_dataset = self.dataset_cls(self.hparams, mode='test')

        if stage in ('predict', None):
            self.predict_dataset = self.dataset_cls(self.hparams, mode='predict')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )

    # def val_dataloader(self):
    #     ...
    #
    # def test_dataloader(self):
    #     ...
    #

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=False
        )
