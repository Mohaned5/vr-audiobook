# fsdp_strategy.py
from lightning.pytorch.strategies import FSDPStrategy
from torch import nn
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

class CustomFSDPStrategy(FSDPStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define your custom auto_wrap_policy
        self.auto_wrap_policy = size_based_auto_wrap_policy(
            min_num_params=1e6  # Adjust threshold based on model size
        )

    @staticmethod
    def get_auto_wrap_policy():
        def policy(module, recurse, _, param_numel):
            return isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer))
        return policy
