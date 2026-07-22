"""CLCFI: Cue-Level Counterfactual Intervention for multimodal sentiment analysis."""

from .config import CLCFIConfig
from .model import CLCFI, CLCFIOutput

__all__ = ["CLCFI", "CLCFIConfig", "CLCFIOutput"]
