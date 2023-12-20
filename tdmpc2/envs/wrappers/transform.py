
from typing import Optional, Callable

import numpy as np

import torch

import warnings
from tensordict import unravel_key, unravel_key_list
from tensordict._tensordict import _unravel_key_to_tuple
from tensordict.nn import dispatch
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right, NestedKey
from torch import nn, Tensor
from torchrl.envs import RandomCropTensorDict, Transform, Compose

class MaskedRandomCropTensorDict(RandomCropTensorDict):
    def __init__(
        self,
        sub_seq_len: int,
        sample_dim: int = -1,
        mask_key: Optional[NestedKey] = None,
        mask_predicate: Optional[Callable] = None,
    ):
        super().__init__(sub_seq_len, sample_dim)
        self.mask_predicate = mask_predicate

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        shape = tensordict.shape
        dim = self.sample_dim
        # shape must have at least one dimension
        if not len(shape):
            raise RuntimeError(
                "Cannot sub-sample from a tensordict with an empty shape."
            )
        if shape[dim] < self.sub_seq_len:
            raise RuntimeError(
                f"Cannot sample trajectories of length {self.sub_seq_len} along"
                f" dimension {dim} given a tensordict of shape "
                f"{tensordict.shape}. Consider reducing the sub_seq_len "
                f"parameter or increase sample length."
            )
        max_idx_0 = shape[dim] - self.sub_seq_len
        idx_shape = list(tensordict.shape)
        idx_shape[dim] = 1
        device = tensordict.device
        if device is None:
            device = torch.device("cpu")
        if self.mask_predicate is None and (self.mask_key is None or self.mask_key not in tensordict.keys(
            isinstance(self.mask_key, tuple))):
            idx_0 = torch.randint(max_idx_0, idx_shape, device=device)
        else:
            if self.mask_predicate is not None:
                mask = self.mask_predicate(tensordict)
            else:
                # get the traj length for each entry
                mask = tensordict.get(self.mask_key)
            if mask.shape != tensordict.shape:
                raise ValueError(
                    "Expected a mask of the same shape as the tensordict. Got "
                    f"mask.shape={mask.shape} and tensordict.shape="
                    f"{tensordict.shape} instead."
                )
            traj_lengths = mask.cumsum(self.sample_dim).max(self.sample_dim, True)[0]
            if (traj_lengths < self.sub_seq_len).any():
                raise RuntimeError(
                    f"Cannot sample trajectories of length {self.sub_seq_len} when the minimum "
                    f"trajectory length is {traj_lengths.min()}."
                )
            # take a random number between 0 and traj_lengths - self.sub_seq_len
            idx_0 = (
                torch.rand(idx_shape, device=device) * (traj_lengths - self.sub_seq_len)
            ).to(torch.long)
        arange = torch.arange(self.sub_seq_len, device=idx_0.device)
        arange_shape = [1 for _ in range(tensordict.ndimension())]
        arange_shape[dim] = len(arange)
        arange = arange.view(arange_shape)
        idx = idx_0 + arange
        return tensordict.gather(dim=self.sample_dim, index=idx)


