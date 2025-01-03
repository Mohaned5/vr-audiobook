from lightning.pytorch.strategies import FSDPStrategy
import torch.nn as nn

class CustomFSDPStrategy(FSDPStrategy):
    def __init__(self, **kwargs):
        super().__init__(
            auto_wrap_policy={nn.Linear, nn.Conv2d},
            sharding_strategy="FULL_SHARD",
            mixed_precision=True,
            activation_checkpointing_policy={nn.TransformerEncoderLayer},
            cpu_offload=False,
            **kwargs
        )
