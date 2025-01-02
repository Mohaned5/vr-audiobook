import os
import torch
import wandb
from models import *
from dataset import *
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
from datetime import timedelta
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM


def cli_main():
    # remove slurm env vars due to this issue:
    # https://github.com/Lightning-AI/lightning/issues/5225
    if 'SLURM_NTASKS' in os.environ:
        del os.environ["SLURM_NTASKS"]
    if 'SLURM_JOB_NAME' in os.environ:
        del os.environ["SLURM_JOB_NAME"]

    torch.set_float32_matmul_precision('medium')

    wandb_id = os.environ.get('WANDB_RUN_ID', wandb.util.generate_id())
    exp_dir = os.path.join('logs', wandb_id)
    os.makedirs(exp_dir, exist_ok=True)
    wandb_logger = lazy_instance(
        WandbLogger,
        project='panfusion',
        id=wandb_id,
        save_dir=exp_dir
        )

    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        train_time_interval=timedelta(minutes=10),
        )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    class MyLightningCLI(LightningCLI):
        def before_instantiate_classes(self):
            # set result_dir, data and pano_height for evaluation
            if self.config.get('test', {}).get('model', {}).get('class_path') == 'models.EvalPanoGen':
                if self.config.test.data.init_args.result_dir is None:
                    result_dir = os.path.join(exp_dir, 'test')
                    self.config.test.data.init_args.result_dir = result_dir
                self.config.test.model.init_args.data = self.config.test.data.class_path.split('.')[-1]
                self.config.test.model.init_args.pano_height = self.config.test.data.init_args.pano_height
                self.config.test.data.init_args.batch_size = 1
        

        def add_arguments_to_parser(self, parser):
            parser.add_argument("--model.init_args.enable_peft", type=bool, default=False, help="Enable PEFT")
            parser.add_argument("--model.init_args.peft_config", type=dict, default={}, help="PEFT configuration")
            parser.link_arguments("model.init_args.cam_sampler", "data.init_args.cam_sampler")

        def before_fit(self):
            # Apply PEFT to the PanFusion model before training
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                task_type="CAUSAL_LM"
            )

            # Access the PanFusion model instance
            pano_model = self.model
            pano_model.instantiate_model()

            # Apply PEFT
            self.model.mv_base_model = get_peft_model(pano_model.mv_base_model, lora_config)
            self.model.mv_base_model.print_trainable_parameters()



    cli = MyLightningCLI(
        trainer_class=Trainer,
        save_config_kwargs={'overwrite': True},
        parser_kwargs={'parser_mode': 'omegaconf', 'default_env': True},
        seed_everything_default=os.environ.get("LOCAL_RANK", 0),
        trainer_defaults={
            'strategy': 'ddp',
            'log_every_n_steps': 10,
            'num_sanity_val_steps': 0,
            'limit_val_batches': 4,
            'benchmark': True,
            'max_epochs': 10,
            'precision': 16,
            'callbacks': [checkpoint_callback, lr_monitor],
            'logger': wandb_logger
        })


if __name__ == '__main__':
    cli_main()
