from lightning.pytorch.strategies import FSDPStrategy
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import torch

mixed_precision_config = MixedPrecision(
    param_dtype=torch.float32,  # Use FP16 for parameters
    reduce_dtype=torch.float32,  # Use FP16 for gradients
    buffer_dtype=torch.float32   # Use FP16 for buffers
)

class CustomFSDPStrategy(FSDPStrategy):
    def __init__(self, **kwargs):
        super().__init__(
            auto_wrap_policy={nn.Linear, nn.Conv2d},
            sharding_strategy="FULL_SHARD",
            # mixed_precision=mixed_precision_config,  # Use the configured object
            activation_checkpointing_policy={nn.TransformerEncoderLayer},
            cpu_offload=False,
            **kwargs
        )
