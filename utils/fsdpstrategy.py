# fsdp_strategy.py
from lightning.pytorch.strategies import FSDPStrategy
from torch import nn
from importlib import import_module
import logging

logger = logging.getLogger(__name__)

class CustomFSDPStrategy(FSDPStrategy):
    def __init__(self, *args, activation_checkpointing_policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if activation_checkpointing_policy:
            self.activation_checkpointing_policy = self._resolve_activation_checkpointing_policy(activation_checkpointing_policy)
            logger.info(f"Activation checkpointing policy set to: {self.activation_checkpointing_policy}")
        else:
            self.activation_checkpointing_policy = None
        
        # Define the auto_wrap_policy
        self.auto_wrap_policy = self.get_auto_wrap_policy()
        logger.info(f"Auto wrap policy defined: {self.auto_wrap_policy}")
    
    @staticmethod
    def get_auto_wrap_policy():
        # Example: Wrap Transformer layers
        def policy(module, recurse, _, param_numel):
            return isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer))
        return policy
    
    @staticmethod
    def _resolve_activation_checkpointing_policy(policy_list):
        resolved = set()
        for class_path in policy_list:
            module_path, class_name = class_path.rsplit('.', 1)
            module = import_module(module_path)
            cls = getattr(module, class_name)
            resolved.add(cls)
        return resolved
