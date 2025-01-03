from lightning.pytorch.strategies import FSDPStrategy
from torch.nn import Module
from typing import Set
import logging

logger = logging.getLogger(__name__)

class CustomFSDPStrategy(FSDPStrategy):
    def __init__(self, *args, target_modules: Set[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_modules = target_modules

        # Define the custom auto-wrap policy
        self.auto_wrap_policy = self.get_auto_wrap_policy()

        logger.info(f"Using custom auto-wrap policy for target modules: {self.target_modules}")

    def get_auto_wrap_policy(self):
        target_modules = self.target_modules

        def policy(module: Module, recurse: bool, _, param_numel: int):
            # Match the module's fully qualified name against the target modules
            for name, _ in module.named_modules():
                if any(name.endswith(target) for target in target_modules):
                    return True
            return False

        return policy
