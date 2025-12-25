"""
Model architectures for SMART-IQA and HyperIQA.
"""

from .smart_iqa import (
    MultiScaleAttention,
    HyperNet,
    TargetNet,
    TargetFC,
    SwinBackbone
)

from .hyperiqa import (
    HyperNet as HyperIQA_HyperNet,
    TargetNet as HyperIQA_TargetNet
)

__all__ = [
    'MultiScaleAttention',
    'HyperNet',
    'TargetNet',
    'TargetFC',
    'SwinBackbone',
    'HyperIQA_HyperNet',
    'HyperIQA_TargetNet',
]
