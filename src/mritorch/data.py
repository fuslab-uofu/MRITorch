"""
In-memory containers for k-space and images.

Layouts are C-order with "loop" dimensions leading and the spatial block
trailing, so library functions can act on ``(..., coil, z, y, x)``:

- `KSpace`: ``(slice, set, phase, contrast, repetition, average, coil, kz, ky, kx)``,
  sorted from ISMRMRD `EncodingCounters` (mirrors Gadgetron's buffered data).
- `Image`: ``(..., [coil,] z, y, x)`` with names in `dims`.

Both carry dimension names so singleton loop dimensions can be dropped with
`squeeze()` without losing track of what each axis means.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import torch

from mritorch.geometry import Geometry

LOOP_DIMS = ("slice", "set", "phase", "contrast", "repetition", "average")
KSPACE_DIMS = LOOP_DIMS + ("coil", "kz", "ky", "kx")
IMAGE_DIMS = LOOP_DIMS + ("coil", "z", "y", "x")
SPATIAL_K = ("kz", "ky", "kx")
SPATIAL_I = ("z", "y", "x")


def _squeeze(data: torch.Tensor, dims: tuple[str, ...], keep: Sequence[str]):
    drop = [i for i, (n, s) in enumerate(zip(dims, data.shape)) if s == 1 and n not in keep]
    for i in reversed(drop):
        data = data.squeeze(i)
    return data, tuple(n for i, n in enumerate(dims) if i not in drop)


class _Named:
    data: torch.Tensor
    dims: tuple[str, ...]
    _keep: tuple[str, ...] = ()

    def dim(self, name: str) -> int:
        """Axis index of the dimension called `name`."""
        return self.dims.index(name)

    def size(self, name: str) -> int:
        """Length of the dimension called `name`, or 1 if absent."""
        return self.data.shape[self.dims.index(name)] if name in self.dims else 1

    def squeeze(self):
        """Drop singleton loop dimensions (spatial and coil dims are kept)."""
        data, dims = _squeeze(self.data, self.dims, self._keep)
        return replace(self, data=data, dims=dims)  # type: ignore

    def permute(self, order: Sequence[str]):
        """Reorder dimensions by name (port of MATLAB `get_permutation`)."""
        if sorted(order) != sorted(self.dims):
            raise ValueError(f"order {tuple(order)} must be a permutation of {self.dims}")
        perm = [self.dims.index(n) for n in order]
        return replace(self, data=self.data.permute(perm), dims=tuple(order))  # type: ignore

    def to(self, *args, **kwargs):
        """`torch.Tensor.to` applied to the data; the trajectory follows its device."""
        changes: dict[str, Any] = {"data": self.data.to(*args, **kwargs)}
        traj = getattr(self, "traj", None)
        if traj is not None:
            changes["traj"] = traj.to(device=changes["data"].device)
        return replace(self, **changes)  # type: ignore

    @property
    def shape(self) -> dict[str, int]:
        return dict(zip(self.dims, self.data.shape))


@dataclass
class KSpace(_Named):
    """
    Sorted k-space data.

    Attributes
    ----------
    data : torch.Tensor
        Complex k-space, by default with dims `KSPACE_DIMS`.
    header : ismrmrd.xsd.ismrmrdHeader or None
        Parsed ISMRMRD XML header.
    dims : tuple of str
        Name of each axis of `data`.
    traj : torch.Tensor or None
        Non-Cartesian trajectory, shape ``(..., kz, ky, kx, D)`` matching
        `data` without the coil axis.
    noise : torch.Tensor or None
        Noise-scan samples, shape ``(coil, samples)``.
    geometry : dict
        `Geometry` per slice index, from the acquisition headers.
    sample_time_us : float or None
        Readout dwell time.
    """
    data: torch.Tensor
    header: Any = None
    dims: tuple[str, ...] = KSPACE_DIMS
    traj: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None
    geometry: dict[int, Geometry] = field(default_factory=dict)
    sample_time_us: Optional[float] = None
    _keep: tuple[str, ...] = field(default=("coil",) + SPATIAL_K, repr=False)


@dataclass
class Image(_Named):
    """
    Image data with geometry.

    Attributes
    ----------
    data : torch.Tensor
        Image tensor, trailing dims ``(z, y, x)``.
    geometry : Geometry or None
        Grid geometry of the trailing ``(z, y, x)`` block. For multislice 2D
        data, use `slice_geometry` instead.
    dims : tuple of str
        Name of each axis of `data`.
    header : ismrmrd.xsd.ismrmrdHeader or None
        ISMRMRD XML header of the source data.
    slice_geometry : dict
        `Geometry` per slice index, for 2D multislice data.
    meta : dict
        Free-form metadata (TE, TI, units, ...).
    """
    data: torch.Tensor
    geometry: Optional[Geometry] = None
    dims: tuple[str, ...] = IMAGE_DIMS
    header: Any = None
    slice_geometry: dict[int, Geometry] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    _keep: tuple[str, ...] = field(default=SPATIAL_I, repr=False)

    @property
    def affine(self) -> torch.Tensor:
        """Voxel-index -> LPS-mm affine of the ``(z, y, x)`` block."""
        if self.geometry is None:
            raise ValueError("Image has no geometry")
        return self.geometry.affine
