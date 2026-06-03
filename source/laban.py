"""Computable, continuous LMA-inspired style descriptors.

The four descriptors -- V (vertical), H (horizontal), P (pace), and R
(regularity) -- are defined in Section 3.2 of the paper. V and H are
geometric and operate on joint positions; P operates on per-joint speeds;
R compares predicted and ground-truth joint rotations.
"""

import torch


class LabanDescriptors:
    """Compute the V, H, P, R descriptors for a fixed skeleton.

    The ``skeleton`` dictionary maps role-based joint names to joint
    indices, decoupling the descriptor formulas from any particular
    dataset's joint ordering. Required keys are ``head``, ``neck``,
    ``hips``, ``spine``, ``chest``, ``shoulders``, ``hands``,
    ``upper_arms``, ``lower_arms``, ``feet``, ``upper_legs`` and
    ``lower_legs``; paired joints (e.g. ``hands``) take 2-element lists.
    """

    def __init__(self, skeleton: dict, device: str = 'cpu') -> None:
        self.device = device
        self.skeleton = skeleton

    # ---- public descriptors ------------------------------------------------

    def vertical(self, positions: torch.Tensor) -> torch.Tensor:
        """V descriptor: mean angle over the 11 specified joint triplets.

        Args:
            positions: Joint positions of shape (B, T, J, 3).

        Returns:
            Tensor of shape (B,).
        """
        return self._triplet_angles(positions).mean(dim=(-1, -2))

    def horizontal(self, positions: torch.Tensor) -> torch.Tensor:
        """H descriptor: mean L2 distance over the 13 specified joint pairs.

        Args:
            positions: Joint positions of shape (B, T, J, 3).

        Returns:
            Tensor of shape (B,).
        """
        return self._pair_distances(positions).mean(dim=(-1, -2))

    def pace(self, positions: torch.Tensor) -> torch.Tensor:
        """P descriptor: per-joint average speed across all joints and time.

        Args:
            positions: Joint positions of shape (B, T, J, 3).

        Returns:
            Tensor of shape (B,).
        """
        return self._joint_speeds(positions).mean(dim=(-1, -2, -3))

    def regularity(self,
                   pred_rotations: torch.Tensor,
                   gt_rotations: torch.Tensor,
                   lambda_: float = 1.0,
                   epsilon: float = 1e-7) -> torch.Tensor:
        """R descriptor: geodesic-distance regularity term.

        Implements the manuscript formula (Section 3.2.4)::

            R = (1 / (T * N_J)) * sum_t sum_j d_geo(R^pred_j(t), R^gt_j(t))
                + lambda * (1 / T) * sum_t sigma_J(t),

        where ``sigma_J(t)`` is the standard deviation of the per-joint
        geodesic distances at time ``t``.

        Args:
            pred_rotations: Predicted local rotation matrices, shape
                (B, T, J, 3, 3).
            gt_rotations: Ground-truth local rotation matrices, shape
                (B, T, J, 3, 3).
            lambda_: Weight on the per-time joint-wise std term (manuscript: 1).
            epsilon: Numerical guard for the acos clamp.

        Returns:
            Tensor of shape (B,).
        """
        relative = torch.matmul(pred_rotations, gt_rotations.transpose(-1, -2))
        trace = torch.diagonal(relative, offset=0, dim1=-2, dim2=-1).sum(-1)
        ratio = torch.clamp((trace - 1.0) / 2.0, -1.0 + epsilon, 1.0 - epsilon)
        geodesic = torch.acos(ratio)  # (B, T, J)

        mean_over_t_j = geodesic.mean(dim=(-1, -2))            # (B,)
        std_over_j = geodesic.std(dim=-1, unbiased=False)      # (B, T)
        mean_std = std_over_j.mean(dim=-1)                     # (B,)
        return mean_over_t_j + lambda_ * mean_std

    # ---- internals ---------------------------------------------------------

    def _triplet_angles(self, positions: torch.Tensor) -> torch.Tensor:
        """11 angle triplets from Section 3.2.1. Returns shape (B, T, 11)."""
        skel = self.skeleton
        triplets = [
            self._angle(positions, skel["hands"][0], skel["lower_arms"][0], skel["upper_arms"][0]),
            self._angle(positions, skel["hands"][1], skel["lower_arms"][1], skel["upper_arms"][1]),
            self._angle(positions, skel["feet"][0], skel["lower_legs"][0], skel["upper_legs"][0]),
            self._angle(positions, skel["feet"][1], skel["lower_legs"][1], skel["upper_legs"][1]),
            self._angle(positions, skel["head"], skel["neck"], skel["spine"]),
            self._angle(positions, skel["chest"], skel["spine"], skel["hips"]),
            self._angle(positions, skel["lower_arms"][0], skel["upper_arms"][0], skel["shoulders"][0]),
            self._angle(positions, skel["lower_arms"][1], skel["upper_arms"][1], skel["shoulders"][1]),
            self._angle(positions, skel["hands"][0], skel["hips"], skel["hands"][1]),
            self._angle(positions, skel["upper_arms"][0], skel["hips"], skel["upper_arms"][1]),
            self._angle(positions, skel["feet"][0], skel["hips"], skel["feet"][1]),
        ]
        return torch.concat(triplets, dim=-1)

    def _angle(self,
               positions: torch.Tensor,
               start_joint: int,
               center_joint: int,
               end_joint: int,
               epsilon: float = 1e-7) -> torch.Tensor:
        """Angle (in radians) at ``center_joint`` between bones to start and end."""
        v1 = positions[:, :, end_joint] - positions[:, :, center_joint]
        v2 = positions[:, :, start_joint] - positions[:, :, center_joint]

        norm1 = torch.maximum(torch.norm(v1, dim=-1, keepdim=True, p=2),
                              torch.tensor(epsilon, device=self.device))
        norm2 = torch.maximum(torch.norm(v2, dim=-1, keepdim=True, p=2),
                              torch.tensor(epsilon, device=self.device))
        cos = torch.sum((v1 / norm1) * (v2 / norm2), dim=-1, keepdim=True)
        cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)
        return torch.acos(cos)

    def _pair_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """13 L2 pair distances from Section 3.2.2. Returns shape (B, T, 13)."""
        skel = self.skeleton
        pairs = [
            self._pair_distance(positions, *skel["hands"]),
            self._pair_distance(positions, *skel["lower_arms"]),
            self._pair_distance(positions, *skel["upper_arms"]),
            self._pair_distance(positions, *skel["feet"]),
            self._pair_distance(positions, *skel["lower_legs"]),
            self._pair_distance(positions, *skel["upper_legs"]),
            self._pair_distance(positions, skel["head"], skel["hands"][0]),
            self._pair_distance(positions, skel["head"], skel["hands"][1]),
            self._pair_distance(positions, skel["hips"], skel["head"]),
            self._pair_distance(positions, skel["hips"], skel["hands"][0]),
            self._pair_distance(positions, skel["hips"], skel["hands"][1]),
            self._pair_distance(positions, skel["lower_arms"][0], skel["hands"][1]),
            self._pair_distance(positions, skel["lower_arms"][1], skel["hands"][0]),
        ]
        return torch.concat(pairs, dim=-1)

    def _pair_distance(self,
                       positions: torch.Tensor,
                       joint_a: int,
                       joint_b: int) -> torch.Tensor:
        """L2 distance between two joints."""
        diff = positions[:, :, joint_a] - positions[:, :, joint_b]
        return torch.norm(diff, p=2, dim=-1, keepdim=True)

    def _joint_speeds(self, positions: torch.Tensor) -> torch.Tensor:
        """Per-joint speeds with the initial frame's speed pinned to zero.

        Returns shape (B, T, J, 1).
        """
        squared = torch.sum(torch.square(torch.diff(positions, dim=1, prepend=positions[:, 0:1])), dim=-1)
        return torch.sqrt(torch.maximum(squared, torch.tensor(1e-6, device=self.device)))[..., None]
