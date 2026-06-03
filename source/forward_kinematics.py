"""Forward kinematics on a fixed skeleton.

This module exposes :class:`ForwardKinematics`, which walks a parent
dictionary to convert per-joint local rotations into world-space joint
positions and rotations. It also exposes small convenience wrappers around
:mod:`source.rotations` for the BVH ZYX Euler convention used by this project.
"""

from typing import List, Optional

import bvhio
import torch

from .rotations import (
    euler_angles_to_matrix, matrix_to_euler_angles,
    matrix_to_quaternion, quaternion_to_matrix,
    matrix_to_rotation_6d, rotation_6d_to_matrix,
)


_ROTATION_FORMATS = ('euler', 'quaternion', 'sixd', 'matrix')


class ForwardKinematics:
    """Forward-kinematics solver for a fixed skeleton."""

    def __init__(self, parent: dict, offsets: torch.Tensor, device: str = 'cpu') -> None:
        self.device = device
        self.parent = parent
        self.offsets = offsets.to(device=self.device, dtype=torch.float32)
        self.num_joints = self.offsets.shape[0]

        self.positions: Optional[torch.Tensor] = None
        self.rotations: Optional[torch.Tensor] = None
        self.root_translations: Optional[torch.Tensor] = None
        self.intrinsic_matrices: Optional[torch.Tensor] = None
        self.original_shape: Optional[tuple] = None
        self.flat_batch_size: int = 0

    @classmethod
    def from_bvh(cls,
                 hierarchy_file: Optional[str] = None,
                 bvh_container: Optional[bvhio.BvhContainer] = None,
                 device: str = 'cpu') -> 'ForwardKinematics':
        """Construct a ForwardKinematics solver from a BVH file or container."""
        if bvh_container is None and hierarchy_file is None:
            raise ValueError("Either hierarchy_file or bvh_container must be provided.")
        if bvh_container is None:
            bvh_container = bvhio.readAsBvh(hierarchy_file)

        layout = bvh_container.Root.layout()

        offsets_list = [joint.Offset.to_list() for joint, _, _ in layout]
        offsets = torch.tensor(offsets_list, dtype=torch.float32)

        name_to_id = {joint.Name: index for joint, index, _ in layout}
        parent_dict = {}
        for joint, index, _ in layout:
            for child in joint.Children:
                parent_dict[name_to_id[child.Name]] = index

        return cls(parent=parent_dict, offsets=offsets, device=device)

    def set_rotations(self,
                      rotations: torch.Tensor,
                      rotation_format: str,
                      root_translations: Optional[torch.Tensor] = None) -> None:
        """Set the local rotations and run the FK pass.

        Args:
            rotations: Local rotations of shape (*batch_dims, num_joints, N),
                where N depends on ``rotation_format``.
            rotation_format: One of ``{'euler', 'quaternion', 'sixd', 'matrix'}``.
            root_translations: Optional root translations of shape
                (*batch_dims, 3). When omitted, the root sits at its rest offset.
        """
        if rotation_format not in _ROTATION_FORMATS:
            raise ValueError(f"Unknown rotation format: {rotation_format!r}")

        self.original_shape = rotations.shape[:-2]

        if root_translations is not None:
            if root_translations.shape[:-1] != self.original_shape:
                raise ValueError(
                    f"Shape mismatch: root_translations {root_translations.shape} "
                    f"is incompatible with rotations {rotations.shape}"
                )
            self.root_translations = root_translations.reshape(-1, 3).to(self.device)
        else:
            self.root_translations = None

        if rotation_format == 'euler':
            rotations = euler_to_matrix(rotations)
        elif rotation_format == 'quaternion':
            rotations = quat_to_matrix(rotations)
        elif rotation_format == 'sixd':
            rotations = sixd_to_matrix(rotations)
        # matrix: no conversion needed.

        rotations = rotations.reshape(-1, self.num_joints, 3, 3)
        self.flat_batch_size = rotations.size(0)
        self.intrinsic_matrices = rotations.transpose(0, 1).to(self.device)
        self._calculate_kinematics()

    def get_positions(self, joint_ids: Optional[List[int]] = None) -> torch.Tensor:
        """Return the calculated world-space joint positions."""
        if self.positions is None:
            raise RuntimeError("Positions not calculated. Call set_rotations() first.")

        positions = self.positions.transpose(0, 1)  # (batch, joints, 3)

        if joint_ids is not None:
            positions = positions[:, joint_ids]
            joint_count = len(joint_ids)
        else:
            joint_count = self.num_joints

        return positions.reshape(self.original_shape + (joint_count, 3))

    def get_rotations(self,
                      flat: bool,
                      world_space: bool,
                      rotation_format: str,
                      joint_ids: Optional[List[int]] = None) -> torch.Tensor:
        """Return the calculated rotations in the requested format."""
        if self.rotations is None or self.intrinsic_matrices is None:
            raise RuntimeError("Rotations not calculated. Call set_rotations() first.")
        if rotation_format not in _ROTATION_FORMATS:
            raise ValueError(f"Unknown rotation format: {rotation_format!r}")

        if world_space:
            rotations = self.rotations.transpose(0, 1)
        else:
            rotations = self.intrinsic_matrices.transpose(0, 1)

        rotations = rotations.reshape(self.original_shape + (self.num_joints, 3, 3))

        if joint_ids is not None:
            rotations = rotations[..., joint_ids, :, :]

        if rotation_format == 'euler':
            rotations = matrix_to_euler(rotations)
        elif rotation_format == 'quaternion':
            rotations = matrix_to_quat(rotations)
        elif rotation_format == 'sixd':
            rotations = matrix_to_sixd(rotations)

        if flat:
            rotations = rotations.reshape(self.original_shape + (-1,))

        return rotations

    def _calculate_kinematics(self) -> None:
        """In-place FK pass, used by :meth:`set_rotations`."""
        self.positions = torch.zeros(self.num_joints, self.flat_batch_size, 3, device=self.device)
        self.rotations = torch.zeros(self.num_joints, self.flat_batch_size, 3, 3, device=self.device)

        root_base = self.offsets[0].expand(self.flat_batch_size, 3)
        if self.root_translations is not None:
            self.positions[0] = root_base + self.root_translations
        else:
            self.positions[0] = root_base
        self.rotations[0] = self.intrinsic_matrices[0]

        for joint_idx in range(1, self.num_joints):
            parent_idx = self.parent[joint_idx]
            parent_rot = self.rotations[parent_idx]
            offset = self.offsets[joint_idx].expand(self.flat_batch_size, 3).unsqueeze(2)
            self.positions[joint_idx] = self.positions[parent_idx] + torch.matmul(parent_rot, offset).squeeze(-1)
            self.rotations[joint_idx] = torch.matmul(parent_rot, self.intrinsic_matrices[joint_idx])

    def compute(self,
                rotations: torch.Tensor,
                root_translations: Optional[torch.Tensor] = None):
        """Differentiable stateless FK pass.

        Args:
            rotations: Local rotation matrices of shape (*batch_dims, num_joints, 3, 3).
            root_translations: Optional root translations of shape (*batch_dims, 3).

        Returns:
            Tuple ``(world_positions, world_rotations)`` shaped
            (*batch_dims, num_joints, 3) and (*batch_dims, num_joints, 3, 3).
        """
        original_shape = rotations.shape[:-3]
        num_joints = rotations.shape[-3]

        flat_rotations = rotations.reshape(-1, num_joints, 3, 3)
        flat_batch_size = flat_rotations.size(0)
        intrinsic = flat_rotations.transpose(0, 1)

        world_positions = [None] * num_joints
        world_rotations = [None] * num_joints

        root_base = self.offsets[0].expand(flat_batch_size, 3)
        if root_translations is not None:
            flat_root = root_translations.reshape(flat_batch_size, 3)
            world_positions[0] = root_base + flat_root
        else:
            world_positions[0] = root_base
        world_rotations[0] = intrinsic[0]

        for joint_idx in range(1, self.num_joints):
            parent_idx = self.parent[joint_idx]
            parent_pos = world_positions[parent_idx]
            parent_rot = world_rotations[parent_idx]
            offset = self.offsets[joint_idx].expand(flat_batch_size, 3).unsqueeze(2)
            world_positions[joint_idx] = parent_pos + torch.matmul(parent_rot, offset).squeeze(-1)
            world_rotations[joint_idx] = torch.matmul(parent_rot, intrinsic[joint_idx])

        positions = torch.stack(world_positions, dim=0).transpose(0, 1).reshape(original_shape + (num_joints, 3))
        rotations_world = torch.stack(world_rotations, dim=0).transpose(0, 1).reshape(original_shape + (num_joints, 3, 3))
        return positions, rotations_world


# --- ZYX Euler convention shims around source.rotations -----------------------

_EULER_CONVENTION = 'ZYX'


def matrix_to_euler(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Matrix to ZYX Euler angles, wrapped into the (-pi, pi] range."""
    euler = matrix_to_euler_angles(rotation_matrix, _EULER_CONVENTION)
    return torch.where(euler > torch.pi, euler - 2 * torch.pi, euler)


def euler_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    return euler_angles_to_matrix(euler, _EULER_CONVENTION)


def matrix_to_quat(rotation_matrix: torch.Tensor) -> torch.Tensor:
    return matrix_to_quaternion(rotation_matrix)


def quat_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(quaternion)


def matrix_to_sixd(rotation_matrix: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(rotation_matrix)


def sixd_to_matrix(sixd: torch.Tensor) -> torch.Tensor:
    return rotation_6d_to_matrix(sixd)
