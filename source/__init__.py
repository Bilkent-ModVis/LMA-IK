"""Data-driven inverse kinematics using Laban Movement Analysis."""

from .dataset import (
    MotionDataset,
    build_bandai_dataset,
    build_dance_dataset,
    build_lma_effort_dataset,
)
from .forward_kinematics import ForwardKinematics
from .interpolator import Interpolator
from .laban import LabanDescriptors
from .synthesizer import Synthesizer

__all__ = [
    "ForwardKinematics",
    "Interpolator",
    "LabanDescriptors",
    "MotionDataset",
    "Synthesizer",
    "build_bandai_dataset",
    "build_dance_dataset",
    "build_lma_effort_dataset",
]

__version__ = "0.1.0"
