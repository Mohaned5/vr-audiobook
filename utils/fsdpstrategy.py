# fsdp_strategy.py
from lightning.pytorch.strategies import FSDPStrategy
from torch import nn

class CustomFSDPStrategy(FSDPStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define your custom auto_wrap_policy
        self.auto_wrap_policy = self.get_auto_wrap_policy()

    @staticmethod
    def get_auto_wrap_policy():
        # Example: Wrap Transformer layers
        def policy(module, recurse, _, param_numel):
            return isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer))
        
        return policy
