"""Compatibility helpers for the pinned LIBERO/robosuite stack."""
from __future__ import annotations

import os


# Small CPU SVDs are much slower with the server's default BLAS thread count.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy
import torch


def apply_libero_compat() -> None:
    """Allow pinned LIBERO state files and fix NumPy/enum membership."""
    torch.serialization.add_safe_globals(
        [
            numpy.core.multiarray._reconstruct,
            numpy.ndarray,
            numpy.dtype,
            numpy.dtypes.Float64DType,
        ]
    )

    import mujoco
    from robosuite.utils.binding_utils import MjModel

    def get_joint_qpos_addr(self, name):
        joint_id = int(self.joint_name2id(name))
        joint_type = int(self.jnt_type[joint_id])
        joint_addr = int(self.jnt_qposadr[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            ndim = 7
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            ndim = 4
        else:
            assert joint_type in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
            ndim = 1
        return joint_addr if ndim == 1 else (joint_addr, joint_addr + ndim)

    def get_joint_qvel_addr(self, name):
        joint_id = int(self.joint_name2id(name))
        joint_type = int(self.jnt_type[joint_id])
        joint_addr = int(self.jnt_dofadr[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            ndim = 6
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            ndim = 3
        else:
            assert joint_type in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
            ndim = 1
        return joint_addr if ndim == 1 else (joint_addr, joint_addr + ndim)

    MjModel.get_joint_qpos_addr = get_joint_qpos_addr
    MjModel.get_joint_qvel_addr = get_joint_qvel_addr
