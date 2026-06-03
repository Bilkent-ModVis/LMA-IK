"""BVH dataset construction and access.

This module ingests BVH motion-capture files, normalizes the skeleton
height-wise, slices the motion into fixed-length training windows, and
caches per-window V/H/P style descriptors. The R descriptor is computed
at training time against ground-truth rotations, so it is stored as a
zero placeholder here and its normalization range is fixed to [0, pi].
"""

import math
from pathlib import Path
from typing import Iterable, List, Optional

import bvhio
import torch
from tqdm import tqdm

from .forward_kinematics import ForwardKinematics
from .laban import LabanDescriptors


TRAINING_DATA_KEYS: List[str] = [
    'angles_6d', 'positions_sites',
    'laban_v', 'laban_h', 'laban_p', 'laban_r',
]


class MotionDataset:
    """Dataset of fixed-length motion windows with cached LMA descriptors.

    Each entry in the dataset is a dict with keys

    - ``angles_6d``: local joint rotations in 6D representation,
      shape ``(seq_len, num_joints, 6)``.
    - ``positions_sites``: end-effector positions, shape
      ``(seq_len, num_sites, 3)``.
    - ``laban_v``, ``laban_h``, ``laban_p``: per-window V, H, P values,
      normalized to ``[0, 1]``. Shape ``()``.
    - ``laban_r``: zero-placeholder for R (computed at training time
      from predicted vs. ground-truth rotations).
    """

    def __init__(self,
                 dataset_root: Path,
                 laban_skeleton: dict,
                 site_ids: List[int],
                 root_index: int = 0,
                 sequence_length: int = 50,
                 max_frame_diff: int = 5,
                 device: str = 'cpu',
                 excludes: Optional[Iterable[Path]] = None,
                 do_processing: bool = True) -> None:
        self.dataset_root = Path(dataset_root)
        self.device = device
        self.sequence_length = sequence_length
        self.max_frame_diff = max_frame_diff
        self.root_index = root_index
        self.laban_skeleton = laban_skeleton
        self.site_ids = site_ids

        self.parent: dict = {}
        self.offsets: Optional[torch.Tensor] = None
        self.num_joints: int = 0
        self.converter: Optional[ForwardKinematics] = None
        self.laban: Optional[LabanDescriptors] = None

        self.training_data = {key: [] for key in TRAINING_DATA_KEYS}
        self.laban_limits = {'v': [], 'h': [], 'p': [], 'r': []}
        self.position_limits: Optional[dict] = None

        if do_processing and self.dataset_root.exists():
            self.process(excludes)

    # ------------------------------------------------------------------
    # Build phase
    # ------------------------------------------------------------------

    def process(self, excludes: Optional[Iterable[Path]] = None) -> None:
        self._initialize_helpers()
        bvh_files = self._find_bvh_files(excludes)
        buffers = {key: [] for key in TRAINING_DATA_KEYS}

        print("Processing BVH files and creating sequences...")
        for bvh_path in tqdm(bvh_files, desc="Processing files"):
            full_motion = self._process_single_bvh(bvh_path)
            if full_motion is None:
                continue

            for window in self._slice_motion_into_sequences(full_motion):
                features = self._calculate_laban_features(window['positions'])
                for key in buffers:
                    if key.startswith('laban_'):
                        buffers[key].append(features[key.split('_')[1]])
                    elif key in window:
                        buffers[key].append(window[key])

        print("Finalizing dataset...")
        if not buffers['angles_6d']:
            print("Warning: No sequences were created.")
            return

        self.training_data = buffers
        self._calculate_position_limits()
        self._normalize_laban_features()
        print(f"Processing complete. Created {len(self)} sequences.")

    def _initialize_helpers(self) -> None:
        self.normalized_skeleton, self.height_norm_factor = build_normalized_skeleton(self.dataset_root)
        self.num_joints = len(self.normalized_skeleton.Root.layout())
        self.converter = ForwardKinematics.from_bvh(bvh_container=self.normalized_skeleton, device=self.device)
        self.parent = self.converter.parent
        self.offsets = self.converter.offsets.cpu()
        self.laban = LabanDescriptors(self.laban_skeleton, device=self.device)

    # ------------------------------------------------------------------
    # Per-BVH processing
    # ------------------------------------------------------------------

    def _find_bvh_files(self, excludes: Optional[Iterable[Path]] = None) -> List[Path]:
        bvh_files = list(self.dataset_root.rglob('*.bvh'))
        if excludes:
            resolved = {Path(ex).resolve() for ex in excludes}
            bvh_files = [f for f in bvh_files if f.resolve() not in resolved]
        return bvh_files

    def _process_single_bvh(self, bvh_path: Path):
        try:
            bvh = bvhio.readAsBvh(bvh_path)
        except Exception as exc:
            print(f"Warning: Could not read {bvh_path}. Error: {exc}")
            return None

        layout = bvh.Root.layout()
        rotations_quat = torch.tensor(
            [[joint[0].Keyframes[frame].Rotation.to_list() for joint in layout]
             for frame in range(bvh.FrameCount)],
            dtype=torch.float32,
        )
        root_translations = torch.tensor(
            [layout[self.root_index][0].Keyframes[frame].Position.to_list()
             for frame in range(bvh.FrameCount)],
            dtype=torch.float32,
        )
        root_translations /= self.height_norm_factor
        self.converter.set_rotations(rotations_quat, 'quaternion', root_translations=root_translations)

        return {
            'angles_6d': self.converter.get_rotations(flat=False, world_space=False, rotation_format='sixd').cpu(),
            'positions': self.converter.get_positions(),
            'positions_sites': self.converter.get_positions(self.site_ids),
        }

    def _slice_motion_into_sequences(self, motion):
        num_frames = motion['angles_6d'].shape[0]
        for stride in range(1, self.max_frame_diff + 1):
            for start in range(0, num_frames - self.sequence_length * stride, 1):
                indices = torch.arange(start, start + self.sequence_length * stride, stride)
                yield {key: value[indices] for key, value in motion.items()}

    # ------------------------------------------------------------------
    # Descriptors and normalization
    # ------------------------------------------------------------------

    def _calculate_laban_features(self, positions_seq: torch.Tensor) -> dict:
        # Add a leading batch axis so descriptors return a length-1 vector.
        pos = positions_seq.unsqueeze(0)
        features = {
            'v': self.laban.vertical(pos).cpu().squeeze(0),
            'h': self.laban.horizontal(pos).cpu().squeeze(0),
            'p': self.laban.pace(pos).cpu().squeeze(0),
            # R is computed from predicted vs ground-truth rotations at
            # training time (Section 3.2.4); the target is randomized,
            # so the stored value is a placeholder.
            'r': torch.zeros((), dtype=torch.float32),
        }
        for key, val in features.items():
            if key == 'r':
                continue
            value_min = val.min()
            value_max = val.max()
            if not self.laban_limits[key]:
                self.laban_limits[key] = [value_min, value_max]
            else:
                self.laban_limits[key][0] = torch.min(self.laban_limits[key][0], value_min)
                self.laban_limits[key][1] = torch.max(self.laban_limits[key][1], value_max)
        return features

    def _normalize_laban_features(self) -> None:
        print("Normalizing Laban features...")
        # R values are bounded by [0, pi] (mean geodesic + per-time std).
        # Fix the range so normalized R targets sit in [0, 1] regardless
        # of dataset statistics (Section 3.2.4).
        self.laban_limits['r'] = [torch.tensor(0.0), torch.tensor(float(math.pi))]
        for key in ('v', 'h', 'p', 'r'):
            limits = self.laban_limits[key]
            if not limits or (limits[1] - limits[0]) < 1e-6:
                continue
            data_key = f'laban_{key}'
            self.training_data[data_key] = [
                (tensor - limits[0]) / (limits[1] - limits[0])
                for tensor in self.training_data[data_key]
            ]

    def _calculate_position_limits(self) -> None:
        print("Calculating global position limits for normalization...")
        global_min = torch.tensor(float('inf'))
        global_max = torch.tensor(float('-inf'))
        for tensor in self.training_data['positions_sites']:
            global_min = torch.min(global_min, tensor.min())
            global_max = torch.max(global_max, tensor.max())
        self.position_limits = {'min': global_min, 'max': global_max}
        print(f"Global position min value: {global_min.item()}")
        print(f"Global position max value: {global_max.item()}")

    def denormalize_positions(self, normalized: torch.Tensor) -> torch.Tensor:
        if self.position_limits is None:
            raise ValueError("Position limits not available. Cannot de-normalize.")
        lo = self.position_limits['min'].to(normalized.device)
        hi = self.position_limits['max'].to(normalized.device)
        span = hi - lo
        if span < 1e-6:
            span = torch.tensor(1.0, device=normalized.device)
        return (normalized + 1) / 2 * span + lo

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.training_data['angles_6d']) if self.training_data.get('angles_6d') else 0

    def __getitem__(self, index: int) -> dict:
        if index >= len(self):
            raise IndexError("Index out of range")
        return {key: self.training_data[key][index] for key in self.training_data}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            'training_data': self.training_data,
            'laban_limits': self.laban_limits,
            'position_limits': self.position_limits,
            'parent': self.parent,
            'offsets': self.offsets,
            'num_joints': self.num_joints,
            'laban_skeleton': self.laban_skeleton,
            'site_ids': self.site_ids,
            'sequence_length': self.sequence_length,
            'max_frame_diff': self.max_frame_diff,
        }
        torch.save(bundle, path)
        print(f"Dataset saved to {path}")

    @classmethod
    def load(cls,
             path: Path,
             device: str = 'cpu',
             load_training_data: bool = True) -> 'MotionDataset':
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        bundle = torch.load(path, map_location='cpu')

        instance = cls(
            dataset_root=Path('.'),
            laban_skeleton=bundle['laban_skeleton'],
            site_ids=bundle['site_ids'],
            sequence_length=bundle.get('sequence_length', 50),
            max_frame_diff=bundle.get('max_frame_diff', 5),
            device=device,
            do_processing=False,
        )

        instance.laban_limits = bundle['laban_limits']
        instance.position_limits = bundle['position_limits']
        instance.num_joints = bundle['num_joints']
        instance.parent = bundle['parent']
        instance.offsets = bundle['offsets']

        instance.converter = ForwardKinematics(parent=instance.parent, offsets=instance.offsets, device=device)
        instance.laban = LabanDescriptors(instance.laban_skeleton, device=device)

        if load_training_data:
            instance.training_data = {
                k: [t.to(device) for t in tensors]
                for k, tensors in bundle['training_data'].items()
            }
            print(f"Dataset fully loaded to '{device}' from {path}. Found {len(instance)} sequences.")
        return instance


# ---------------------------------------------------------------------------
# Helpers and dataset-specific factories
# ---------------------------------------------------------------------------

def build_normalized_skeleton(dataset_root: Path):
    """Pick an arbitrary BVH under ``dataset_root`` and normalize its skeleton
    to unit height. Returns the modified container and the scale factor.
    """
    dataset_root = Path(dataset_root)
    bvh_files = list(dataset_root.rglob("*.bvh"))
    skeleton_bvh = bvhio.readAsBvh(bvh_files[0], loadKeyFrames=False)

    temp_hierarchy = bvhio.convertBvhToHierarchy(skeleton_bvh.Root)
    temp_hierarchy.loadRestPose()
    min_y, max_y = float('inf'), float('-inf')
    for joint, _, _ in temp_hierarchy.layout():
        min_y = min(min_y, joint.PositionWorld.y)
        max_y = max(max_y, joint.PositionWorld.y)

    height = max_y - min_y
    if height < 1e-6:
        height = 1.0

    for joint, _, _ in skeleton_bvh.Root.layout():
        joint.Offset /= height
        if joint.EndSite is not None:
            joint.EndSite /= height

    skeleton_bvh.FrameCount = 0
    return skeleton_bvh, height


def build_lma_effort_dataset(save_path: Path,
                             dataset_root: Path,
                             device: str = 'cpu',
                             sequence_length: int = 50,
                             max_frame_diff: int = 5) -> MotionDataset:
    """Build the LMA Effort dataset (Kim et al., ACM TAP 2022)."""
    laban_skeleton = {
        "head": 5, "neck": 4, "hips": 0, "spine": 1, "chest": 2,
        "shoulders": [6, 10], "hands": [9, 13],
        "upper_arms": [7, 11], "lower_arms": [8, 12],
        "feet": [20, 16], "upper_legs": [18, 14], "lower_legs": [19, 15],
    }
    dataset_root = Path(dataset_root)
    excludes = [
        dataset_root / 'put' / 'subject 8_neutral.bvh',
        dataset_root / 'walk' / 'subject 8_neutral.bvh',
        dataset_root / 'sit down' / 'subject 8_neutral.bvh',
        dataset_root / 'wave' / 'subject 8_neutral.bvh',
    ]
    site_ids = [9, 13, 17, 21]
    dataset = MotionDataset(
        dataset_root=dataset_root,
        laban_skeleton=laban_skeleton,
        site_ids=site_ids,
        sequence_length=sequence_length,
        device=device,
        excludes=excludes,
        max_frame_diff=max_frame_diff,
    )
    dataset.save(save_path)
    return dataset


def build_bandai_dataset(save_path: Path,
                         dataset_root: Path,
                         device: str = 'cpu',
                         sequence_length: int = 50,
                         max_frame_diff: int = 5) -> MotionDataset:
    """Build the Bandai-Namco Research Motion Dataset 2 (Kobayashi et al., 2023)."""
    laban_skeleton = {
        "head": 5, "neck": 4, "hips": 1, "spine": 2, "chest": 3,
        "shoulders": [6, 10], "hands": [9, 13],
        "upper_arms": [7, 11], "lower_arms": [8, 12],
        "feet": [16, 20], "upper_legs": [14, 18], "lower_legs": [15, 19],
    }
    site_ids = [9, 13, 17, 21]
    dataset = MotionDataset(
        dataset_root=Path(dataset_root),
        laban_skeleton=laban_skeleton,
        site_ids=site_ids,
        sequence_length=sequence_length,
        device=device,
        excludes=None,
        max_frame_diff=max_frame_diff,
        root_index=1,
    )
    dataset.save(save_path)
    return dataset


def build_dance_dataset(save_path: Path,
                        dataset_root: Path,
                        device: str = 'cpu',
                        sequence_length: int = 50,
                        max_frame_diff: int = 5) -> MotionDataset:
    """Build the Folk Dance Motion Capture dataset (Aristidou et al., 2015)."""
    laban_skeleton = {
        "head": 16, "neck": 14, "hips": 0, "spine": 12, "chest": 13,
        "shoulders": [17, 24], "hands": [20, 27],
        "upper_arms": [18, 25], "lower_arms": [19, 26],
        "feet": [4, 9], "upper_legs": [2, 7], "lower_legs": [3, 8],
    }
    site_ids = [5, 10, 21, 28]
    dataset = MotionDataset(
        dataset_root=Path(dataset_root),
        laban_skeleton=laban_skeleton,
        site_ids=site_ids,
        sequence_length=sequence_length,
        device=device,
        max_frame_diff=max_frame_diff,
    )
    dataset.save(save_path)
    return dataset
