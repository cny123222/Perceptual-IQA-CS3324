"""
SMART-IQA: Swin Multi-scale Attention-guided Regression Transformer for BIQA

A state-of-the-art blind image quality assessment method that achieves
0.9378 SRCC on KonIQ-10k.

Author: Nuoyan Chen
Institution: Shanghai Jiao Tong University
"""

__version__ = "1.0.0"
__author__ = "Nuoyan Chen"

from .models.smart_iqa import SwinBackbone as SmartIQA
from .solvers.smart_solver import SmartIQASolver
from .solvers.hyper_solver import HyperIQASolver

__all__ = [
    'SmartIQA',
    'SmartIQASolver',
    'HyperIQASolver',
]
